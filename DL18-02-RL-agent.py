import abc
import numpy as np
import tensorflow as tf

# pip install tf-agents==0.14

from tf_agents.agents import tf_agent
from tf_agents.drivers import driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step

from DL18_library import BanditPyEnvironment

# we need to have an environment that is a bit harder to figure out.
# To spice up things just a little bit, the next environment will either always give
# reward = observation * action or reward = -observation * action.
# This will be decided when the environment is initialized.

class TwoWayPyEnvironment(BanditPyEnvironment):

  def __init__(self):
    action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
    observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=-2, maximum=2, name='observation')

    # Flipping the sign with probability 1/2.
    self._reward_sign = 2 * np.random.randint(2) - 1
    print("reward sign:")
    print(self._reward_sign)

    super(TwoWayPyEnvironment, self).__init__(observation_spec, action_spec)

  def _observe(self):
    self._observation = np.random.randint(-2, 3, (1,), dtype='int32')
    return self._observation

  def _apply_action(self, action):
    return self._reward_sign * action * self._observation[0]

two_way_tf_environment = tf_py_environment.TFPyEnvironment(TwoWayPyEnvironment())

nest = tf.nest


# Test sur 1 coup
environment = TwoWayPyEnvironment()
observation = environment.reset().observation
print("observation: %d" % observation)

action = 2

print("action: %d" % action)
reward = environment.step(action).reward
print("reward: %f" % reward)

# TFEnvironment
tf_environment = tf_py_environment.TFPyEnvironment(environment)

# A more complicated environment calls for a more complicated policy.
# We need a policy that detects the behavior of the underlying environment.
# There are three situations that the policy needs to handle:
#
# The agent has not detected know yet which version of the environment is running.
# The agent detected that the original version of the environment is running.
# The agent detected that the flipped version of the environment is running.


class TwoWaySignPolicy(tf_policy.TFPolicy):
  def __init__(self, situation):
    observation_spec = tensor_spec.BoundedTensorSpec(
        shape=(1,), dtype=tf.int32, minimum=-2, maximum=2)
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(), dtype=tf.int32, minimum=0, maximum=2)
    time_step_spec = ts.time_step_spec(observation_spec)
    self._situation = situation
    super().__init__(time_step_spec=time_step_spec,
                                           action_spec=action_spec)
  def _distribution(self, time_step):
    pass

  def _variables(self):
    return [self._situation]

  def _action(self, time_step, policy_state, seed):
    sign = tf.cast(tf.sign(time_step.observation[0, 0]), dtype=tf.int32)
    def case_unknown_fn():
      # Choose 1 so that we get information on the sign.
      return tf.constant(1, shape=(1,))

    # Choose 0 or 2, depending on the situation and the sign of the observation.
    def case_normal_fn():
      return tf.constant(sign + 1, shape=(1,))
    def case_flipped_fn():
      return tf.constant(1 - sign, shape=(1,))

    cases = [(tf.equal(self._situation, 0), case_unknown_fn),
             (tf.equal(self._situation, 1), case_normal_fn),
             (tf.equal(self._situation, 2), case_flipped_fn)]
    action = tf.case(cases, exclusive=True)
    return policy_step.PolicyStep(action, policy_state)

  # Agent
  # Now it's time to define the agent that detects the sign of the environment and sets the policy appropriately.

class SignAgent(tf_agent.TFAgent):
  def __init__(self):
    self._situation = tf.Variable(0, dtype=tf.int32)
    policy = TwoWaySignPolicy(self._situation)
    time_step_spec = policy.time_step_spec
    action_spec = policy.action_spec
    super().__init__(time_step_spec=time_step_spec,
                                    action_spec=action_spec,
                                    policy=policy,
                                    collect_policy=policy,
                                    train_sequence_length=None)

  def _initialize(self):
    return tf.compat.v1.variables_initializer(self.variables)

  def _train(self, experience, weights=None):
    observation = experience.observation
    action = experience.action
    reward = experience.reward

    # We only need to change the value of the situation variable if it is
    # unknown (0) right now, and we can infer the situation only if the
    # observation is not 0.
    needs_action = tf.logical_and(tf.equal(self._situation, 0),
                                  tf.not_equal(reward, 0))

    def new_situation_fn():
      """This returns either 1 or 2, depending on the signs."""
      return (3 - tf.sign(tf.cast(observation[0, 0, 0], dtype=tf.int32) *
                          tf.cast(action[0, 0], dtype=tf.int32) *
                          tf.cast(reward[0, 0], dtype=tf.int32))) / 2

    new_situation = tf.cond(needs_action,
                            new_situation_fn,
                            lambda: self._situation)
    new_situation = tf.cast(new_situation, tf.int32)
    tf.compat.v1.assign(self._situation, new_situation)
    return tf_agent.LossInfo((), ())

sign_agent = SignAgent()

# Trajectories
# n TF-Agents, trajectories are named tuples that contain samples from previous steps taken.
# These samples are then used by the agent to train and update the policy.
# In RL, trajectories must contain information about the current state, the next state, and whether the
# current episode has ended. Since in the Bandit world we do not need these things, we set up a helper
# function to create a trajectory:

# We need to add another dimension here because the agent expects the
# trajectory of shape [batch_size, time, ...], but in this tutorial we assume
# that both batch size and time are 1. Hence all the expand_dims.

def trajectory_for_bandit(initial_step, action_step, final_step):
  return trajectory.Trajectory(observation=tf.expand_dims(initial_step.observation, 0),
                               action=tf.expand_dims(action_step.action, 0),
                               policy_info=action_step.info,
                               reward=tf.expand_dims(final_step.reward, 0),
                               discount=tf.expand_dims(final_step.discount, 0),
                               step_type=tf.expand_dims(initial_step.step_type, 0),
                               next_step_type=tf.expand_dims(final_step.step_type, 0))

# Training

step = two_way_tf_environment.reset()
for _ in range(10):
  action_step = sign_agent.collect_policy.action(step)
  next_step = two_way_tf_environment.step(action_step.action)
  experience = trajectory_for_bandit(step, action_step, next_step)
  print(experience)
  sign_agent.train(experience)
  step = next_step

# From the output one can see that after the second step (unless the observation was 0 in the first step), the policy chooses the action in the right way and thus the reward collected is always non-negative.