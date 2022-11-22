# standard libaries
import argparse
import shutil
from pathlib import Path

# third party libraries
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = MODEL_DIR / "logs/"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"


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
        with open(PREPARED_DIR / f"{target}.joblib", "rb") as fp:
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


def _make_dataset(sentences, max_len, words2index, tags2index):
    y = [[tags2index.get(w[-1], tags2index.get("O")) for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tags2index["O"])
    X = [[words2index.get(w[0], words2index.get("PADword")) for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=words2index["PADword"])

    x_tensor = tf.convert_to_tensor(X)
    y_tensor = tf.convert_to_tensor(y)

    ds = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
    return ds.batch(32)


def main(split: str) -> None:
    """Main Function"""
    # Training variables
    split = split
    episodes = 1_000
    max_len = 100

    # Setup for RLNER
    log_dir = LOG_DIR / f"{split}"
    train_sentences = _get_data(split)
    val_sentences = _get_data("validation")
    words = _get_words(train_sentences)
    tags = _get_tags(train_sentences)
    words2index = _to_idx(words)
    tags2index = _to_idx(tags)

    train_ds = _make_dataset(train_sentences, max_len, words2index, tags2index)
    val_ds = _make_dataset(val_sentences, max_len, words2index, tags2index)

    # Prepare Logs
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True)

    if any(log_dir.iterdir()):
        for i in log_dir.glob("**/*"):
            if i.is_dir():
                shutil.rmtree(i)
            else:
                i.unlink()

    # Instansiate Model
    model1, model2 = create_model(
        vocab_size=len(words2index), max_length=max_len, embedding_dim=100, word_index=words2index, tag_index=tags2index
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )

    _ = model1.fit(
        train_ds,
        epochs=100,
        verbose=1,
        validation_data=val_ds,
        callbacks=[callback, tensorboard_callback],
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
        max_len=max_len,
    )

    # seq tag env
    env = SeqTagEnv(data_pool.labels(), reward_function=reward_fn, observation_featurizer=observation_featurizer)
    for sample, weight in data_pool:
        env.add_sample(sample, weight)

    # Define agent
    agent = Agent(action_dim=len(data_pool.labels()))

    # Train Agent
    train(agent, env, episodes, render=False)

    # Save final agent
    with open(MODEL_DIR / f"agents/agent_{split}.joblib", "wb") as fp:
        joblib.dump(agent, fp, compress=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train_Sequnce_Tag",
        description="Train a REINFORCE agent to label sequences",
    )

    parser.add_argument("split")

    args = parser.parse_args()

    main(split=args.split)
