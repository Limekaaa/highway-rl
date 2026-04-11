import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from IPython.display import clear_output
from tqdm import tqdm


def run_one_episode(env: Env, agent, display=True, seed=None):
    display_env = deepcopy(env)
    done = False
    state, _ = display_env.reset(seed=seed)

    rewards = 0
    length = 0

    while not done:
        action = agent.get_action(state, 0)
        state, reward, terminated, truncated, _ = display_env.step(action)
        done = terminated or truncated
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


def eval_agent(
    env: Env,
    agent,
    n_sim: int | None = None,
    seeds: list[int] | None = None,
    show_progress: bool = False,
):
    """
    Monte Carlo evaluation.

    Repeat n_sim times:
        * Run the policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    env_copy = deepcopy(env)

    seeds = seeds if seeds is not None else [None] * n_sim
    episode_rewards = np.zeros(len(seeds))
    episode_lengths = np.zeros(len(seeds))

    bar = tqdm(
        enumerate(seeds),
        desc="Evaluating",
        disable=not show_progress,
        unit="ep",
        total=len(seeds),
    )

    for i, seed in bar:
        episode_rewards[i], episode_lengths[i] = run_one_episode(
            env_copy, agent, display=False, seed=seed
        )
    return episode_rewards, episode_lengths
