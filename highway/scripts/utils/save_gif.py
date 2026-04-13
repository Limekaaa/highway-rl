from copy import deepcopy
from gymnasium import Env
import os
from matplotlib import pyplot as plt
import imageio

def save_gif(env: Env, agent, path: str, seed=None, make_deep_copy=True):
    if make_deep_copy:
        display_env = deepcopy(env)
    else:
        display_env = env
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