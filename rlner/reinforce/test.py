# third party libraries
import gym

# rlner libraries
from rlner.reinforce.agent import Agent
from rlner.reinforce.utils import train

if __name__ == "__main__":
    agent = Agent()
    episodes = 2
    env = gym.make("MountainCar-v0")
    train(agent, env, episodes)
    env.close()
