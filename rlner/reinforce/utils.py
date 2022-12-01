# third party libraries
import numpy as np
from tqdm import tqdm

# rlner libraries
from rlner.nlp_gym.envs.seq_tagging.env import SeqTagEnv
from rlner.reinforce.agent import Agent

MAX_PATIENCE = 50
MIN_EPISODES = 200


def train(agent: Agent, env: SeqTagEnv, episodes: int, render=True) -> None:
    """Train an agent"""
    patience = MAX_PATIENCE
    max_reward = -999

    # Train to convergence or max episodes
    for episode in tqdm(range(episodes), desc="Episode"):
        # Check Convergence
        if patience <= 0:
            break

        samples = env.get_samples()
        average_rewards = []
        for sample in tqdm(samples, desc="Sample"):
            done = False
            state = env.reset(sample)
            total_reward = 0
            rewards = []
            states = []
            actions = []

            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                state = next_state
                total_reward += reward
                if render:
                    env.render()
                if done:
                    agent.learn(states, rewards, actions)
                    average_rewards.append(total_reward)

        if np.mean(average_rewards) > max_reward:
            patience = MAX_PATIENCE
            max_reward = np.mean(average_rewards)
        elif episode >= MIN_EPISODES:
            patience -= 1
        else:
            pass

        mean_rewards = np.mean(average_rewards)
        std_rewards = np.std(average_rewards)

        message = [
            f"Episode #: {episode + 1}",
            f"avg_reward: {mean_rewards:0.6f} +/- {std_rewards:0.6f}",
            f"patience: {patience}",
            f"top_reward: {max_reward:0.6f}",
        ]

        print("\t".join(message), end="\r")


def predict(agent: Agent, env: SeqTagEnv, render: bool = True):
    """Predict labels"""
    output = []
    samples = env.get_samples()
    for sample in tqdm(samples, desc="Sample"):
        done = False
        state = env.reset(sample)
        total_rewards = 0
        rewards = []
        states = []
        actions = []

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            action_as_str = env.action_space.ix_to_action(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action_as_str)
            state = next_state
            total_rewards += reward
            if render:
                env.render()
            if done:
                output.append(actions)

    return output
