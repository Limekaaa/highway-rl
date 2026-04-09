from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from IPython.display import clear_output


def run_one_episode(env: Env, agent, display=True, seed=None):
    display_env = deepcopy(env)
    done = False
    state, _ = display_env.reset(seed=seed)

    rewards = 0
    length = 0

    while not done:
        action = agent.get_action(state, 0)
        state, reward, done, _, _ = display_env.step(action)
        rewards += reward
        length += 1
        if display:
            clear_output(wait=True)
            plt.imshow(display_env.render())
            plt.show()
    if display:
        display_env.close()
    print(f"Episode reward={rewards:.2f}, length={length}")
    return rewards, length


def eval_agent(env: Env, agent, n_sim: int | None = 5, seeds: list[int] | None = None):
    """
    Monte Carlo evaluation.

    Repeat n_sim times:
        * Run the policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    episode_lengths = np.zeros(n_sim)

    seeds = seeds if seeds is not None else [None] * n_sim

    for i, seed in enumerate(seeds):
        state, _ = env_copy.reset(seed=seed)
        reward_sum = 0
        done = False
        while not done:
            action = agent.get_action(state, 0)
            state, reward, terminated, truncated, _ = env_copy.step(action)
            reward_sum += reward
            episode_lengths[i] += 1
            done = terminated or truncated
        episode_rewards[i] = reward_sum
    return episode_rewards, episode_lengths
