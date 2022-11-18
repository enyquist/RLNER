# rlner libraries
from rlner.nlp_gym.data_pools.custom_seq_tagging_pools import Re3dTaggingPool
from rlner.nlp_gym.envs.seq_tagging.env import SeqTagEnv
from rlner.nlp_gym.envs.seq_tagging.reward import EntityF1Score
from rlner.reinforce.agent import Agent
from rlner.reinforce.utils import train


def eval_agent(agent: Agent, env: SeqTagEnv):
    """Eval an agent"""
    done = False
    obs = env.reset()
    total_reward = 0.0
    actions = []
    while not done:
        action = agent.get_action(obs)
        obs, rewards, done, info = env.step(action)
        actions.append(env.action_space.ix_to_action(action))
        total_reward += rewards
    print("---------------------------------------------")
    print(f"Text: {env.current_sample.text}")
    print(f"Predicted Label {actions}")
    print(f"Oracle Label: {env.current_sample.label}")
    print(f"Total Reward: {total_reward}")
    print("---------------------------------------------")


if __name__ == "__main__":
    # Training variables
    episodes = 20000

    # data pool
    data_pool = Re3dTaggingPool.prepare(split=str(0.0))

    # reward function
    reward_fn = EntityF1Score(dense=True, average="micro")

    # seq tag env
    env = SeqTagEnv(data_pool.labels(), reward_function=reward_fn)
    for sample, weight in data_pool:
        env.add_sample(sample, weight)

    # Define agent
    agent = Agent(action_dim=len(data_pool.labels()))

    train(agent, env, episodes, render=False)
    for _ in range(10):
        eval_agent(agent, env)
    env.close()
