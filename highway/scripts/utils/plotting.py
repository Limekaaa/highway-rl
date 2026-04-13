import matplotlib.pyplot as plt
import numpy as np


def plot_losses(losses, nb_smoothen=500, title="Losses during training", x_values=None):
    losses = np.asarray(losses, dtype=float)
    x = np.arange(len(losses), dtype=float) if x_values is None else np.asarray(x_values, dtype=float)
    smooth_losses = np.convolve(losses, np.ones(nb_smoothen) / nb_smoothen, mode="valid")

    # Center the smoothed curve on the original x-axis, including non-uniform steps.
    smooth_centers = np.arange(len(smooth_losses), dtype=float) + (nb_smoothen - 1) / 2.0
    x_smoothen = np.interp(smooth_centers, np.arange(len(x), dtype=float), x)

    plt.plot(x, losses)
    plt.plot(x_smoothen, smooth_losses, color="red")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()


def plot_train_rewards_lengths(rewards, lengths, nb_smoothen=10, title="Rewards and episode lengths during training", x_values=None):
    rewards = np.asarray(rewards, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    x = np.arange(len(rewards), dtype=float) if x_values is None else np.asarray(x_values, dtype=float)

    smooth_rewards = np.convolve(rewards, np.ones(nb_smoothen) / nb_smoothen, mode="valid")
    smooth_lengths = np.convolve(lengths, np.ones(nb_smoothen) / nb_smoothen, mode="valid")
    rewards_per_frame = rewards / lengths
    smooth_rewards_per_frame = np.convolve(rewards_per_frame, np.ones(nb_smoothen) / nb_smoothen, mode="valid")

    # Center smoothed curves on the original x-axis, including non-uniform evaluation timesteps.
    smooth_centers = np.arange(len(smooth_rewards), dtype=float) + (nb_smoothen - 1) / 2.0
    x_smoothen = np.interp(smooth_centers, np.arange(len(x), dtype=float), x)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title)
    axs[0].plot(x, rewards)
    axs[0].plot(x_smoothen, smooth_rewards, color="red")
    axs[0].set_title("Rewards during training")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Mean reward")
    axs[0].set_ylim(0, 30)

    axs[1].plot(x, lengths)
    axs[1].plot(x_smoothen, smooth_lengths, color="red")
    axs[1].set_title("Episode lengths during training")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Mean episode length")
    axs[1].set_ylim(0, 30)

    axs[2].plot(x, rewards_per_frame)
    axs[2].plot(x_smoothen, smooth_rewards_per_frame, color="red")
    axs[2].set_title("Episode reward per frame during training")
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Mean reward per frame")
    axs[2].set_ylim(None, 1)
    plt.tight_layout()
    plt.show()


def plot_rewards_lengths(rewards, lengths, title=None):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)
    axs[0].hist(rewards, bins=20, color="blue", alpha=0.7, density=True)
    axs[0].set_title("Distribution of Rewards")
    axs[0].set_xlabel("Reward")
    axs[0].set_ylabel("Frequency")
    axs[0].set_ylim(0, 1)

    axs[1].hist(lengths, bins=20, color="orange", alpha=0.7, density=True)
    axs[1].set_title("Distribution of Episode Lengths")
    axs[1].set_xlabel("Episode Length")
    axs[1].set_ylabel("Frequency")
    axs[1].set_ylim(0, 1)

    axs[2].scatter(lengths, rewards, alpha=0.8, marker="+")
    axs[2].plot([0, 30], [0, 30], color="red", linestyle="--", alpha=0.5)
    min_1 = -0.02
    min_30 = 29 * (1.5 / 2.2) - 29 * 0.02
    axs[2].plot([1, 30], [min_1, min_30], color="blue", linestyle="--", alpha=0.5)
    axs[2].set_title("Reward vs Episode Length")
    axs[2].set_xlabel("Episode Length")
    axs[2].set_ylabel("Reward")
    axs[2].set_xlim(0, 31)
    axs[2].set_ylim(0, 31)
    plt.tight_layout()
    plt.show()
