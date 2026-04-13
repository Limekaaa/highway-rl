import datetime
import os
import time
from copy import deepcopy
from enum import Enum
from logging import INFO, FileHandler, Formatter, Logger, getLogger

import numpy as np
import torch
from gymnasium import Env
from tqdm import tqdm

from highway.models.dqn.config import DqnConfig, DqnTrainConfig
from highway.models.dqn.dqn import DQN
from highway.scripts.environment import ConfigType, get_env
from highway.scripts.run import eval_agent
from highway.scripts.seed import set_seed
from shared_core_config import SHARED_CORE_CONFIG, TEST_CONFIG


class LoggingMode(Enum):
    NONE = 0
    TQDM = 1
    LOGGING = 2


def train(
    env: Env,
    agent: DQN,
    n_episodes: int,
    eval_every: int,
    n_sim_per_eval: int,
    reward_threshold: int,
    use_tqdm: bool = True,
    logger: Logger = None,
    date_str: str = "",
):
    total_time = 0
    state, _ = env.reset()
    losses, all_rewards, all_lengths = [], [], []
    best_reward, best_model_state = -float("inf"), None
    eval_env = get_env(config_type=ConfigType.SHARED_CORE)
    start = time.time()

    bar = tqdm(range(n_episodes), desc="Training", disable=not use_tqdm, unit="ep")

    try:
        for ep in bar:
            done = False
            state, _ = env.reset()
            while not done:
                action = agent.get_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                loss_val = agent.update(state, action, reward, terminated, next_state)

                state = next_state
                losses.append(loss_val)

                done = terminated or truncated
                total_time += 1

            if ep % eval_every == 0 or ep == n_episodes - 1:
                rewards, lengths = eval_agent(eval_env, agent, n_sim=n_sim_per_eval)
                cur_reward = np.mean(rewards)
                cur_length = np.mean(lengths)
                all_rewards.append(cur_reward)
                all_lengths.append(cur_length)

                if logger:
                    time_per_episode = (time.time() - start) / (
                        eval_every + n_sim_per_eval
                    )
                    start = time.time()
                    logger.info(
                        "Episode %5d: Mean Reward: %.2f, Mean Length: %.2f, Epsilon: %.2f, Time/Ep: %.2fs",
                        ep,
                        cur_reward,
                        cur_length,
                        agent.epsilon,
                        time_per_episode,
                    )

            # Save every 10 times the eval_every
            if (ep + 1) % (10 * eval_every) == 0:
                os.makedirs(os.path.join("model_weights", "dqn"), exist_ok=True)
                # Save current model and optimizer state
                torch.save(
                    {
                        "model_state_dict": agent.q_net.state_dict(),
                        "optimizer_state_dict": agent.optimizer.state_dict(),
                        "episode": ep,
                        "reward": cur_reward,
                    },
                    os.path.join(
                        "model_weights",
                        "dqn",
                        f"dqn_checkpoint{"" if not date_str else f"_{date_str}"}_ep{ep+1}_reward{cur_reward:.2f}.pth",
                    ),
                )

            mean_reward = np.mean(all_rewards[-5:]) if len(all_rewards) > 0 else 0
            mean_length = np.mean(all_lengths[-5:]) if len(all_lengths) > 0 else 0
            bar.set_postfix(
                {
                    "step": agent.n_steps,
                    "ep": agent.n_eps,
                    "eps": agent.epsilon,
                    "reward": mean_reward,
                    "length": mean_length,
                }
            )

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_model_state = deepcopy(agent.q_net.state_dict())

            if cur_reward >= reward_threshold:
                break

    except KeyboardInterrupt:
        message = "Training interrupted by user."
        print(message)
        if logger:
            logger.info(message)

    return losses, all_rewards, all_lengths, best_model_state


def create_logger() -> Logger:
    os.makedirs(os.path.join("logs", "dqn"), exist_ok=True)
    logger = getLogger(f"DQN_Training")
    logger.setLevel(INFO)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    handler = FileHandler(
        os.path.join(
            "logs",
            "dqn",
            f"DQN_Training_{date_str}.log",
        )
    )
    handler.setFormatter(Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return logger, date_str


if __name__ == "__main__":
    logger, date_str = create_logger()

    dqn_config = DqnConfig()
    train_config = DqnTrainConfig()
    seed = 42

    # Write the hyperparameters to the logger
    logger.info("Seed: %d", seed)
    logger.info("DQN Configuration: %s", dqn_config)
    logger.info("Training Configuration: %s", train_config)
    logger.info("Environment Configuration: %s", TEST_CONFIG)

    set_seed(seed)

    # Environment
    env = get_env(config_type=ConfigType.TEST)
    action_space = env.action_space
    observation_space = env.observation_space

    # Agent
    agent = DQN(
        **dqn_config._asdict(),
        action_space=action_space,
        observation_space=observation_space,
    )

    logger.info("Starting training")

    # Train the agent
    losses, rewards, lengths, best_model_state = train(
        env,
        agent,
        use_tqdm=False,
        logger=logger,
        date_str=date_str,
        **train_config._asdict(),
    )

    # Save the best model
    logger.info("Saving best model...")
    if best_model_state is not None:
        os.makedirs(os.path.join("model_weights", "dqn"), exist_ok=True)
        torch.save(
            best_model_state,
            os.path.join(
                "model_weights",
                "dqn",
                f"dqn_best_model_{date_str}.pth",
            ),
        )

    # Save other results
    logger.info("Saving training results...")
    os.makedirs(os.path.join("results", "dqn", "loss"), exist_ok=True)
    os.makedirs(os.path.join("results", "dqn", "reward"), exist_ok=True)
    os.makedirs(os.path.join("results", "dqn", "length"), exist_ok=True)
    np.save(
        os.path.join("results", "dqn", "loss", f"dqn_losses_{date_str}.npy"),
        np.array(losses),
    )
    np.save(
        os.path.join("results", "dqn", "reward", f"dqn_rewards_{date_str}.npy"),
        np.array(rewards),
    )
    np.save(
        os.path.join("results", "dqn", "length", f"dqn_lengths_{date_str}.npy"),
        np.array(lengths),
    )

    logger.info("Training completed.")
