"""
Testing saved models within Ant-v4 environment of mujoco.
"""

import argparse
import torch
import numpy as np
from gym.envs.mujoco.ant_v4 import AntEnv
from parameters import MAX_STEP
from PPO import Ppo
from observation_normalizer import ObservationNormalizer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name", type=str, default="Ant-v4", help="name of Mujoco environement"
)
parser.add_argument(
    "--model_path", type=str, default="./models/", help="where models are saved"
)
parser.add_argument(
    "--episode_num", type=int, default=100, help="How many times to test"
)

args = parser.parse_args()


def run():
    """
    Start testing with given options.
    """

    device = torch.device('cuda:0') if torch.cuda.is_available() \
        else torch.device('cpu')
    print(device)

    env = AntEnv()
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    torch.manual_seed(500)
    np.random.seed(500)

    with open("log_" + args.env_name + ".csv", "a", encoding="utf-8") as outfile:
        outfile.write("episode_id,score\n")

    ppo = Ppo(s_dim, a_dim, device)
    normalizer = ObservationNormalizer(s_dim)
    ppo.actor_net = torch.load(args.model_path + "actor_net.pt")
    ppo.critic_net = torch.load(args.model_path + "critic_net.pt")
    normalizer.load(args.model_path)
    episode_id = 0

    for episode_id in range(args.episode_num):
        now_state = normalizer(env.reset(seed=500))
        score = 0
        for _ in range(MAX_STEP):
            env.render()

            with torch.no_grad():
                    ppo.actor_net.eval()
                    a = ppo.actor_net.choose_action(torch.from_numpy(
                        np.array(now_state).astype(np.float32)).unsqueeze(0).to(device))[0]
            now_state, r, done, _, _ = env.step(a)
            now_state = normalizer(now_state)
            score += r

            if done:
                break
        print("episode: ", episode_id, "\tscore: ", score)


if __name__ == "__main__":
    run()
