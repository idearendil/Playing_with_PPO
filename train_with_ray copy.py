"""
Start training models by PPO method within Ant-v4 environment of mujoco,
using ray package to implement multi-learners operating on cpu cores.
"""

import argparse
from copy import deepcopy
import torch
import numpy as np
from gym.envs.mujoco.ant_v4 import AntEnv
import ray
from parameters import MAX_STEP, CYCLE_NUM, MIN_STEPS_IN_CYCLE, MIN_STEPS_PER_ACTOR, GAMMA, LAMBDA, NUM_CPUS
from PPO import Ppo
from observation_normalizer import ObservationNormalizer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name",
    type=str,
    default="Ant-v4",
    help="name of Mujoco environement"
)
parser.add_argument(
    "--conitinue_training",
    type=bool,
    default=False,
    help="whether to continue training with existing models",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./models/",
    help="where models are saved"
)
args = parser.parse_args()

ray.init(num_cpus=NUM_CPUS, ignore_reinit_error=True)


def get_gae(rewards, masks, values):
    """
    Calculate Generalized Advantage Estimation.
    """
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + GAMMA * running_returns * masks[t]
        running_tderror = (
            rewards[t] + GAMMA * previous_value * masks[t] - values.data[t]
        )
        running_advants = (
            running_tderror + GAMMA * LAMBDA * running_advants * masks[t]
        )

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants
    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def actor(env, actor_net, critic_net, normalizer):
    """
    Actor task function which collect data by actor_net.
    """
    scores = []
    steps_in_cycle = 0
    episode_memory = []
    actor_net.eval()
    critic_net.eval()
    return_data = []

    while steps_in_cycle < MIN_STEPS_PER_ACTOR:
        now_state = normalizer(env.reset(seed=500))
        score = 0
        for _ in range(MAX_STEP):
            steps_in_cycle += 1

            with torch.no_grad():
                state_tensor = torch.from_numpy(np.array(
                    now_state).astype(np.float32)).unsqueeze(0)
                a, a_prob = actor_net.choose_action(state_tensor)

                next_state, r, done, _, _ = env.step(a)
                next_state = normalizer(next_state)

                mask = (1 - done) * 1
                episode_memory.append([now_state, a, r, mask, a_prob])

                score += r
                now_state = next_state

            if done:
                break

        state_lst, action_lst, reward_lst, mask_lst, prob_lst = \
            [], [], [], [], []
        for a_state, a_action, a_reward, a_mask, a_prob in episode_memory:
            state_lst.append(a_state)
            action_lst.append(torch.Tensor(a_action))
            reward_lst.append(a_reward)
            mask_lst.append(a_mask)
            prob_lst.append(torch.Tensor(a_prob))

        states = torch.Tensor(np.array(state_lst, dtype=np.float32))
        rewards = torch.Tensor(np.array(reward_lst, dtype=np.float32))
        masks = torch.Tensor(np.array(mask_lst, dtype=np.float32))

        with torch.no_grad():
            values = critic_net(states)
            returns, advants = get_gae(rewards, masks, values)

        for idx, _ in enumerate(states):
            return_data.append((states[idx],
                                action_lst[idx],
                                advants[idx],
                                returns[idx],
                                prob_lst[idx]))
        episode_memory = []
        scores.append(score)

    return return_data, scores, normalizer


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

    ppo = Ppo(s_dim, a_dim, device)
    central_normalizer = ObservationNormalizer(s_dim)
    if args.conitinue_training:
        ppo.actor_net = torch.load(
            args.model_path + "actor_net.pt").to(device)
        ppo.critic_net = torch.load(
            args.model_path + "critic_net.pt").to(device)
        central_normalizer.load(args.model_path)

    for cycle_id in range(CYCLE_NUM):
        central_scores = []
        ppo.buffer.buffer.clear()               # off-policy? on-policy?
        ppo.models_to_device('cpu')

        normalizer_lst, env_lst = [], []
        for _ in range(NUM_CPUS):
            normalizer_lst.append(deepcopy(central_normalizer))
            env_lst.append(AntEnv())

        return_data, scores, a_normalizer = actor(env_lst[0], ppo.actor_net, ppo.critic_net, normalizer_lst[0])
        for a_data in return_data:
            ppo.buffer.push(a_data)
        central_scores.extend(scores)
        central_normalizer.combine(a_normalizer)

        score_avg = np.mean(central_scores)
        print("cycle: ", cycle_id, "\tscore: ", score_avg)

        ppo.models_to_device(device)
        ppo.train()
        torch.save(ppo.actor_net, args.model_path + "actor_net.pt")
        torch.save(ppo.critic_net, args.model_path + "critic_net.pt")
        central_normalizer.save(args.model_path)


if __name__ == "__main__":
    run()
