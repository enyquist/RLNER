# standard libaries
import argparse
import shutil
from pathlib import Path

# third party libraries
import tensorflow as tf

# rlner libraries
import rlner.utils as utils
from rlner.nlp_gym.data_pools.custom_seq_tagging_pools import Re3dTaggingPool
from rlner.nlp_gym.envs.common.action_space import ActionSpace
from rlner.nlp_gym.envs.seq_tagging.env import SeqTagEnv
from rlner.nlp_gym.envs.seq_tagging.featurizer import BiLSTMFeaturizerForSeqTagging
from rlner.nlp_gym.envs.seq_tagging.reward import EntityF1Score
from rlner.reinforce.agent import Agent
from rlner.reinforce.utils import predict, train

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"
PREPARED_DIR = DATA_DIR / "prepared"
NOISE_DIR = PREPARED_DIR / "noise"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = MODEL_DIR / "logs/"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"


def main(split: str) -> None:
    """Main Function"""
    # Training variables
    split = split
    episodes = 1_000
    max_len = 100

    # Setup for RLNER
    log_dir = LOG_DIR / f"{split}"
    train_sentences = utils._get_data(split)
    val_sentences = utils._get_data("validation")
    test_sentences = utils._get_data("test")
    words = utils._get_words(train_sentences)
    tags = utils._get_tags(train_sentences)
    words2index = utils._to_idx(words)
    tags2index = utils._to_idx(tags)

    train_ds = utils._make_dataset(train_sentences, max_len, words2index, tags2index)
    val_ds = utils._make_dataset(val_sentences, max_len, words2index, tags2index)

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
    model1, model2 = utils.create_model(
        vocab_size=len(words2index), max_length=max_len, embedding_dim=100, word_index=words2index, tag_index=tags2index
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
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
    test_data_pool = Re3dTaggingPool.prepare(split="test")

    # reward function
    reward_fn = EntityF1Score(dense=False, average="micro")

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
    agent.dump(MODEL_DIR / f"agents/agent_{split}")
    model1.save(MODEL_DIR / f"agents/agent_{split}/model1")
    model2.save(MODEL_DIR / f"agents/agent_{split}/model2")

    # seq tag env
    test_env = SeqTagEnv(data_pool.labels(), reward_function=reward_fn, observation_featurizer=observation_featurizer)
    for sample, weight in test_data_pool:
        test_env.add_sample(sample, weight)

    # Eval Agent
    preds = predict(agent, test_env, render=False)

    for pred, sentence in zip(preds, test_sentences):
        assert len(pred) == len(sentence)
        for idx, p in enumerate(pred):
            sentence[idx] = sentence[idx] + (p,)

    with open(MODEL_DIR / f"agents/agent_{split}/preds.txt", "w") as fp:
        for sentence in test_sentences:
            for token, pos, truth, pred in sentence:
                fp.write(f"{token} {pos} {truth} {pred}\n")
            fp.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train_Sequnce_Tag",
        description="Train a REINFORCE agent to label sequences",
    )

    parser.add_argument("split")

    args = parser.parse_args()

    main(split=args.split)
