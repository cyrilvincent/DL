# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base class for Bandit Python environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
from typing import Optional, Text

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.environments import py_environment, tf_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import nest_utils

"""Base class for bandit environments implemented in TensorFlow."""
@six.add_metaclass(abc.ABCMeta)
class BanditTFEnvironment(tf_environment.TFEnvironment):
  """Base class for bandit environments implemented in TensorFlow.

  Subclasses should implement the `_apply_action` and `_observe` methods.

  Example usage with eager mode:
  ```
    # reset() creates the initial time_step and resets the environment.
    time_step = environment.reset()
    for _ in tf.range(num_steps):
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
  ```

  Example usage with graph mode:
  ```
    # current_time_step() creates the initial TimeStep.
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    # Apply the action and return the new TimeStep.
    next_time_step = environment.step(action_step.action)

    sess.run([time_step, action_step, next_time_step])
  ```
  """

  def __init__(
      self,
      time_step_spec: Optional[types.NestedArray] = None,
      action_spec: Optional[types.NestedArray] = None,
      batch_size: Optional[types.Int] = 1,
      name: Optional[Text] = None,
  ):
    """Initialize instances of `BanditTFEnvironment`.

    Args:
      time_step_spec: A `TimeStep` namedtuple containing `TensorSpec`s defining
        the tensors returned by `step()` (step_type, reward, discount, and
        observation).
      action_spec: A nest of BoundedTensorSpec representing the actions of the
        environment.
      batch_size: The batch size expected for the actions and observations.
      name: The name of this environment instance.
    """
    self._reset_called = tf.compat.v2.Variable(
        False, trainable=False, name='reset_called'
    )

    def _variable_from_spec(name, spec):
      full_shape = [batch_size] + spec.shape.as_list()
      if not name:
        name = 'spec_var'
      return common.create_variable(name, 0, shape=full_shape, dtype=spec.dtype)

    paths_and_specs = nest_utils.flatten_with_joined_paths(time_step_spec)
    variables = [
        _variable_from_spec(path, spec) for path, spec in paths_and_specs
    ]
    self._time_step_variables = tf.nest.pack_sequence_as(
        time_step_spec, variables
    )
    self._name = name

    super(BanditTFEnvironment, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        batch_size=batch_size,
    )

  def _update_time_step(self, time_step):
    tf.nest.map_structure(
        lambda var, value: var.assign(value),
        self._time_step_variables,
        time_step,
    )

  @common.function()
  def _current_time_step(self) -> ts.TimeStep:
    def true_fn():
      return tf.nest.map_structure(tf.identity, self._time_step_variables)

    def false_fn():
      current_time_step = self.reset()
      return current_time_step

    return tf.cond(self._reset_called, true_fn, false_fn)

  @common.function
  def _reset(self) -> ts.TimeStep:
    current_time_step = ts.restart(
        self._observe(),
        batch_size=self.batch_size,
        reward_spec=self.time_step_spec().reward,
    )
    tf.compat.v1.assign(self._reset_called, True)
    self._update_time_step(current_time_step)
    return current_time_step

  @common.function
  def _step(self, action: types.NestedArray) -> ts.TimeStep:
    reward = self._apply_action(action)
    current_time_step = ts.termination(self._observe(), reward)
    self._update_time_step(current_time_step)
    return current_time_step

  @abc.abstractmethod
  def _apply_action(self, action: types.NestedArray) -> types.Float:
    """Returns a reward for the given action."""

  @abc.abstractmethod
  def _observe(self) -> types.NestedTensor:
    """Returns an observation."""

  @property
  def name(self) -> Optional[Text]:
    return self._name

# The _observe function returns an observation. Then, the policy chooses an action based on this observation. The _apply_action receives that action as an input, and returns the corresponding reward. These private member functions are called by the functions reset and step, respectively.
class BanditPyEnvironment(py_environment.PyEnvironment):
  """Base class for Bandit Python environments.

  Every bandit Python environment should derive from this class.
  Subclasses need to implement functions _observe() and _apply_action().

  Usage:

  To receive the first observation, the environment's reset() function should be
  called. To take an action, use the step(action) function. The time step
  returned by step(action) will contain the reward and the next observation.
  """

  def __init__(
      self,
      observation_spec: types.NestedArray,
      action_spec: types.NestedArray,
      reward_spec: Optional[types.NestedArray] = None,
      name: Optional[Text] = None,
  ):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._reward_spec = reward_spec
    self._name = name
    super(BanditPyEnvironment, self).__init__()

  def _reset(self) -> ts.TimeStep:
    """Returns a time step containing an observation.

    It should not be overridden by Bandit environment implementations.

    Returns:
      A time step of type FIRST containing an observation.
    """
    return ts.restart(
        self._observe(),
        batch_size=self.batch_size,
        reward_spec=self.reward_spec(),
    )

  def _step(self, action: types.NestedArray) -> ts.TimeStep:
    """Returns a time step containing the reward for the action taken.

    The returning time step also contains the next observation.
    It should not be overridden by bandit environment implementations.

    Args:
      action: The action taken by the Bandit policy.

    Returns:
      A time step of type LAST containing the reward for the action taken and
      the next observation.
    """
    # This step will take an action and return a reward.
    reward = self._apply_action(action)
    return ts.termination(self._observe(), reward)

  def action_spec(self) -> types.NestedArraySpec:
    return self._action_spec

  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def reward_spec(self) -> types.NestedArraySpec:
    return self._reward_spec

  def _empty_observation(self):
    return tf.nest.map_structure(
        lambda x: np.zeros(x.shape, x.dtype), self.observation_spec()
    )

  @abc.abstractmethod
  def _apply_action(self, action: types.NestedArray) -> types.Float:
    """Applies `action` to the Environment and returns the corresponding reward.

    Args:
      action: A value conforming action_spec that will be taken as action in the
        environment.

    Returns:
      A float value that is the reward received by the environment.
    """

  @abc.abstractmethod
  def _observe(self) -> types.NestedArray:
    """Returns an observation."""

  @property
  def name(self) -> Optional[Text]:
    return self._name

# The following class gives a very simple environment for which the observation is a random integer between -2 and 2, there are 3 possible actions (0, 1, 2), and the reward is the product of the action and the observation
