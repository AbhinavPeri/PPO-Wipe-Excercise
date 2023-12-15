import numpy as np
import robosuite
import gymnasium as gym
import torch
from ppo_continuous_action import Agent
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper
from torch import nn
from torch.distributions.normal import Normal


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer
#
# class Agent(nn.Module):
#     def __init__(self, envs):
#         super().__init__()
#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 1), std=1.0),
#         )
#         self.actor_mean = nn.Sequential(
#             layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
#         )
#         self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
#
#     def get_value(self, x):
#         return self.critic(x)
#
#     def get_action_and_value(self, x, action=None):
#         action_mean = self.actor_mean(x)
#         action_logstd = self.actor_logstd.expand_as(action_mean)
#         action_std = torch.exp(action_logstd)
#         probs = Normal(action_mean, action_std)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == '__main__':

    # Create the environment
    def thunk():
        environment = robosuite.make(
            "Wipe",
            robots=["UR5e"],
            controller_configs=load_controller_config(default_controller="OSC_POSITION"),
            has_renderer=True,  # on-screen rendering
            has_offscreen_renderer=False,  # no off-screen rendering
            control_freq=20,  # 20 hz control for applied actions
            horizon=800,  # each episode terminates after 200 steps
            use_object_obs=True,  # no observations needed
            use_camera_obs=False,  # no observations needed
        )

        environment = GymWrapper(environment)
        environment = gym.wrappers.FlattenObservation(environment)  # deal with dm_control's Dict observation space
        environment = gym.wrappers.RecordEpisodeStatistics(environment)
        environment = gym.wrappers.ClipAction(environment)
        # environment = gym.wrappers.NormalizeObservation(environment)
        # environment = gym.wrappers.TransformObservation(environment, lambda obs: np.clip(obs, -10, 10))
        return environment


    env = thunk()
    # Set the camera
    env.viewer.set_camera(camera_id=0)
    envs = gym.vector.SyncVectorEnv(
        [thunk]
    )

    obs, _ = env.reset()
    obs = torch.Tensor(obs)

    # Load the pytorch model stored at a file path and then visualize its performance using the renderer
    checkpoint = torch.load("runs/Wipe__OSC_POSITION, Bigger Replay Buffer, LR=3e-4, No Observation Transform, Smaller Network, More Epochs__1__2023-11-22_11-58-01/OSC_POSITION, Bigger Replay Buffer, LR=3e-4, No Observation Transform, Smaller Network, More Epochs.cleanrl_model")

    agent = Agent(envs)
    agent.load_state_dict(checkpoint)

    # Perform rollouts and render performance using agent
    done = False
    ret = 0.
    while not done:
        # add dimension to obs
        obs = obs[None, :]

        action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs))
        obs, reward, termination, truncation, infos = env.step(action.numpy().squeeze())  # play action
        done = termination or truncation
        env.render()
        ret += reward

    # Close the environment
    env.close()
    print("rollout completed with return {}".format(ret))
