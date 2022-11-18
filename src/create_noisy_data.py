# standard libaries
import logging
import random
from ast import literal_eval
from pathlib import Path

# third party libraries
import joblib
import pandas as pd
from tqdm import tqdm

# rlner libraries
import rlner.noise as noise
from rlner.utils import SentenceGetter

logger = logging.getLogger()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PREPARED_DIR = DATA_DIR / "prepared"
NOISE_DIR = PREPARED_DIR / "noise"


def main() -> None:
    """Main"""
    # Load master data
    df = pd.read_csv(PREPARED_DIR / "master.csv")
    df["tags"] = df["tags"].apply(literal_eval)

    # Gather as sequences
    getter = SentenceGetter(df)
    sentences = getter.sentences

    # Split into train, val, test
    random.Random(42).shuffle(sentences)
    total_sentences = len(sentences)
    val_idx = int(total_sentences * 0.8)
    test_idx = val_idx + int(total_sentences * 0.1)

    train_sentences = sentences[:val_idx]
    val_sentences = sentences[val_idx:test_idx]
    test_sentences = sentences[test_idx:]

    # Save val/test
    NOISE_DIR.mkdir(exist_ok=True)

    with open(PREPARED_DIR / "validation.joblib", "wb") as fp:
        joblib.dump(val_sentences, fp, compress=3)

    with open(PREPARED_DIR / "test.joblib", "wb") as fp:
        joblib.dump(test_sentences, fp, compress=3)

    # Apply noise and save
    noisy_percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for percentage in tqdm(noisy_percentages, desc="Noise Percentages"):
        noisy_sentences = noise.add_noise(train_sentences, percentage)

        with open(NOISE_DIR / f"noise_{percentage}.joblib", "wb") as fp:
            joblib.dump(noisy_sentences, fp, compress=3)

    logger.info("Success")


if __name__ == "__main__":
    main()
