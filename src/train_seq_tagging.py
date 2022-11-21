# standard libaries
from pathlib import Path

# third party libraries
import joblib

# rlner libraries
from rlner.nlp_gym.data_pools.custom_seq_tagging_pools import Re3dTaggingPool
from rlner.nlp_gym.envs.common.action_space import ActionSpace
from rlner.nlp_gym.envs.seq_tagging.env import SeqTagEnv
from rlner.nlp_gym.envs.seq_tagging.featurizer import BiLSTMFeaturizerForSeqTagging
from rlner.nlp_gym.envs.seq_tagging.reward import EntityF1Score
from rlner.reinforce.agent import Agent
from rlner.reinforce.utils import train
from rlner.utils import create_model

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"
PREPARED_DIR = DATA_DIR / "prepared"
NOISE_DIR = PREPARED_DIR / "noise"


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


def _get_data(target: str):
    if target in ["validation", "test"]:
        with open(NOISE_DIR / f"{target}.joblib", "rb") as fp:
            sentences = joblib.load(fp)
        return sentences

    with open(NOISE_DIR / f"noise_{target}.joblib", "rb") as fp:
        sentences = joblib.load(fp)
    return sentences


def _get_words(sentences):
    words = set([item[0] for sublist in sentences for item in sublist])
    words.add("PADword")
    return words


def _get_tags(sentences):
    tags = set([item[-1] for sublist in sentences for item in sublist])
    return tags


def _to_idx(vals):
    return {val: idx for idx, val in enumerate(vals)}


if __name__ == "__main__":
    # Training variables
    split = 0.0
    episodes = 1000

    # Setup for RLNER
    train_sentences = _get_data(split)
    words = _get_words(train_sentences)
    tags = _get_tags(train_sentences)
    words2index = _to_idx(words)
    tags2index = _to_idx(tags)

    model1, model2 = create_model(
        vocab_size=len(words2index), max_length=100, embedding_dim=100, word_index=words2index, tag_index=tags2index
    )

    # data pool
    data_pool = Re3dTaggingPool.prepare(split=str(split))

    # reward function
    reward_fn = EntityF1Score(dense=True, average="micro")

    # featurizer
    observation_featurizer = BiLSTMFeaturizerForSeqTagging(
        action_space=ActionSpace(data_pool.labels()),
        word2index=words2index,
        model=model2,
        max_len=100,
    )

    # seq tag env
    env = SeqTagEnv(data_pool.labels(), reward_function=reward_fn, observation_featurizer=observation_featurizer)
    for sample, weight in data_pool:
        env.add_sample(sample, weight)

    # Define agent
    agent = Agent(action_dim=len(data_pool.labels()))

    train(agent, env, episodes, render=False)
    for _ in range(10):
        eval_agent(agent, env)
    env.close()
