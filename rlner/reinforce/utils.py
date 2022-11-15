# third party libraries
import gym

# rlner libraries
from rlner.reinforce.agent import Agent


def train(agent: Agent, env: gym.Env, episodes: int, render=True):
    """Train an agent"""
    for episode in range(episodes):
        done = False
        state = env.reset()
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
            print(f"Episode #: {episode}\tep_reward: {total_reward}", end="\r")
