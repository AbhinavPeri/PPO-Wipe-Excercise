import robosuite
from robosuite.controllers import load_controller_config
import numpy as np


def get_policy_action(obs):
    # a trained policy could be used here, but we choose a random action
    low, high = env.action_spec
    return np.random.uniform(low, high)


if __name__ == '__main__':

    controller_config = load_controller_config(default_controller="JOINT_VELOCITY")

    env = robosuite.make(
        "Wipe",
        robots=["UR5e"],
        controller_configs=controller_config,
        has_renderer=True,  # on-screen rendering
        has_offscreen_renderer=False,  # no off-screen rendering
        control_freq=20,  # 20 hz control for applied actions
        horizon=200,  # each episode terminates after 200 steps
        use_object_obs=False,  # no observations needed
        use_camera_obs=False,  # no observations needed
    )

    env.viewer.set_camera(camera_id=0)

    # reset the environment to prepare for a rollout
    obs = env.reset()

    done = False
    ret = 0.
    while not done:
        action = get_policy_action(obs)  # use observation to decide on an action
        obs, reward, done, _ = env.step(action)  # play action
        env.render()
        ret += reward
    print("rollout completed with return {}".format(ret))

    env.close()
