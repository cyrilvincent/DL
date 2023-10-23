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

# SimplePyEnvironment a very simple environment for which the observation is a random integer between -2 and 2,
# there are 3 possible actions (0, 1, 2), and the reward is the product of the action and the observation.
class SimplePyEnvironment(BanditPyEnvironment):

  def __init__(self):
    action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
    observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=-2, maximum=2, name='observation')
    super(SimplePyEnvironment, self).__init__(observation_spec, action_spec)

  def _observe(self):
    self._observation = np.random.randint(-2, 3, (1,), dtype='int32')
    return self._observation

  def _apply_action(self, action):
    return action * self._observation

nest = tf.nest


# Test sur 1 coup
environment = SimplePyEnvironment()
observation = environment.reset().observation
print("observation: %d" % observation)

action = 2

print("action: %d" % action)
reward = environment.step(action).reward
print("reward: %f" % reward)

# TFEnvironment
tf_environment = tf_py_environment.TFPyEnvironment(environment)

# Policy : Politique de jeu
class SignPolicy(tf_policy.TFPolicy):
  def __init__(self):
    observation_spec = tensor_spec.BoundedTensorSpec(
        shape=(1,), dtype=tf.int32, minimum=-2, maximum=2)
    time_step_spec = ts.time_step_spec(observation_spec)

    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(), dtype=tf.int32, minimum=0, maximum=2)

    super(SignPolicy, self).__init__(time_step_spec=time_step_spec,
                                     action_spec=action_spec)
  def _distribution(self, time_step):
    pass

  def _variables(self):
    return ()

  def _action(self, time_step, policy_state, seed):
    observation_sign = tf.cast(tf.sign(time_step.observation[0]), dtype=tf.int32)
    action = observation_sign + 1
    return policy_step.PolicyStep(action, policy_state)

# Now we can request an observation from the environment, call the policy to choose an action, then the environment will output the reward:

sign_policy = SignPolicy()

current_time_step = tf_environment.reset()
print('Observation:')
print (current_time_step.observation)
action = sign_policy.action(current_time_step).action
print('Action:')
print (action)
reward = tf_environment.step(action).reward
print('Reward:')
print(reward)

# every time we take a step, we not only receive the reward for the action we took, but also the next observation.
step = tf_environment.reset()
action = 1
next_step = tf_environment.step(action)
reward = next_step.reward
next_observation = next_step.observation
print("Reward: ")
print(reward)
print("Next observation:")
print(next_observation)