# third party libraries
import numpy as np
from tqdm import tqdm

# rlner libraries
from rlner.nlp_gym.envs.seq_tagging.env import SeqTagEnv
from rlner.reinforce.agent import Agent


def train(agent: Agent, env: SeqTagEnv, episodes: int, render=True):
    """Train an agent"""
    for episode in tqdm(range(episodes), desc="Episode"):
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

        print(
            f"Episode #: {episode}\taverage_reward: {np.mean(average_rewards):0.3f} +/- {np.std(average_rewards):0.3f}",
            end="\r",
        )
