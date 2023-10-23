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

from DL18_Env import SimplePyEnvironment

nest = tf.nest

environment = SimplePyEnvironment()
observation = environment.reset().observation
print("observation: %d" % observation)

action = 2 #@param

print("action: %d" % action)
reward = environment.step(action).reward
print("reward: %f" % reward)