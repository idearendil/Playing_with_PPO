"""
Start training models by PPO method within Ant-v4 environment of mujoco.
"""

import argparse
from collections import deque
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym.envs.mujoco.ant_v4 import AntEnv
from parameters import MAX_STEP, CYCLE_NUM, MIN_STEPS_IN_CYCLE
from PPO import Ppo
from observation_normalizer import ObservationNormalizer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name", type=str, default="Ant-v4", help="name of Mujoco environement"
)
parser.add_argument(
    "--conitinue_training",
    type=bool,
    default=False,
    help="whether to continue training with existing models",
)
parser.add_argument(
    "--model_path", type=str, default="./models/", help="where models are saved"
)
args = parser.parse_args()


def run():
    """
    Start training with given options.
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
    if args.conitinue_training:
        ppo.actor_net = torch.load(
            args.model_path + "actor_net.pt").to(device)
        ppo.critic_net = torch.load(
            args.model_path + "critic_net.pt").to(device)
        normalizer.load(args.model_path)
    episode_id = 0

    for cycle_id in range(CYCLE_NUM):
        scores = []
        steps_in_cycle = 0
        episode_memory = []
        while steps_in_cycle < MIN_STEPS_IN_CYCLE:
            episode_id += 1
            now_state = normalizer(env.reset(seed=500))
            score = 0
            for _ in range(MAX_STEP):
                steps_in_cycle += 1

                with torch.no_grad():
                    ppo.actor_net.eval()
                    a = ppo.actor_net.choose_action(torch.from_numpy(
                        np.array(now_state).astype(np.float32)).unsqueeze(0).to(device))[0]
                next_state, r, done, _, _ = env.step(a)
                next_state = normalizer(next_state)

                mask = (1 - done) * 1
                episode_memory.append([now_state, a, r, mask])

                score += r
                now_state = next_state

                if done:
                    break

            ppo.push_an_episode(episode_memory)
            episode_memory = []

            with open(
                "log_" + args.env_name + ".csv", "a", encoding="utf-8"
            ) as outfile:
                outfile.write(str(episode_id) + "," + str(score) + "\n")
            scores.append(score)
        score_avg = np.mean(scores)
        print("cycle: ", cycle_id, "\tepisode: ", episode_id, "\tscore: ", score_avg)

        ppo.train()
        torch.save(ppo.actor_net, args.model_path + "actor_net.pt")
        torch.save(ppo.critic_net, args.model_path + "critic_net.pt")
        normalizer.save(args.model_path)

    log_df = pd.read_csv("log_" + args.env_name + ".csv")
    plt.plot(log_df["episode_id"], log_df["score"])
    plt.show()


if __name__ == "__main__":
    run()
