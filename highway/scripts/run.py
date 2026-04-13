import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import imageio
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from IPython.display import clear_output
from tqdm import tqdm


def run_one_episode(env: Env, agent, display=True, seed=None, make_deep_copy=True):
    if make_deep_copy:
        display_env = deepcopy(env)
    else :
        display_env = env
        
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
    make_deep_copy: bool = True,
):
    """
    Monte Carlo evaluation.

    Repeat n_sim times:
        * Run the policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    if make_deep_copy:
        env_copy = deepcopy(env)
    else :
        env_copy = env
        
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
            env_copy, agent, display=False, seed=seed, make_deep_copy=make_deep_copy
        )
    return episode_rewards, episode_lengths


def save_gif(env: Env, agent, path: str, seed=None):
    display_env = deepcopy(env)
    done = False
    state, _ = display_env.reset(seed=seed)

    frames = []
    while not done:
        action = agent.get_action(state, 0)
        state, reward, terminated, truncated, _ = display_env.step(action)
        done = terminated or truncated
        frames.append(display_env.render())
    display_env.close()

    # Ensure there are exactly 30 frames by repeating the last frame if necessary
    if len(frames) < 30:
        last_frame = frames[-1]
        frames += [last_frame] * (30 - len(frames))
    elif len(frames) > 30:
        frames = frames[:30]

    # Save the first frame as a PNG (optional, as per your original code)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.figure(figsize=(frames[0].shape[1] / 100, frames[0].shape[0] / 100))
    plt.axis("off")
    plt.imshow(frames[0])
    plt.savefig(path.replace(".gif", ".png"))
    plt.close()

    # Save the GIF with infinite loop and fixed duration
    imageio.mimwrite(
        path,
        frames,
        fps=3,
        loop=0,  # 0 means infinite loop
        duration=1000,  # Fixed duration per frame in milliseconds (adjust as needed)
    )
    os.remove(path.replace(".gif", ".png"))
