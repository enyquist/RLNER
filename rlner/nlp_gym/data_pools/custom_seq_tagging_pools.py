# standard libaries
from pathlib import Path

# third party libraries
import joblib
from flair import datasets
from flair.data import Sentence
from torchnlp.datasets import ud_pos_dataset
from tqdm import tqdm

# rlner libraries
from rlner.nlp_gym.data_pools.base import Sample
from rlner.nlp_gym.data_pools.multi_label_pool import MultiLabelPool

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
PREPARED_DIR = DATA_DIR / "prepared"
NOISE_DIR = PREPARED_DIR / "noise"


class Re3dTaggingPool(MultiLabelPool):
    """Tagging Pool for Re3d Dataset"""

    @classmethod
    def prepare(cls, split: str):
        """Prepare Tagging Pool"""
        sentences = Re3dTaggingPool._get_dataset_from_path(split)

        samples = []
        all_labels = []
        for sent in sentences:
            token_texts = [tok[0] for tok in sent]
            token_texts_ = " ".join(token_texts)

            # check token to text
            flair_sentence = Sentence(token_texts_, use_tokenizer=False)
            assert len(flair_sentence.tokens) == len(token_texts)

            token_labels = [tok[-1] for tok in sent]
            sample = Sample(input_text=token_texts_, oracle_label=token_labels)
            all_labels.extend(token_labels)
            samples.append(sample)
        weights = [1.0] * len(samples)
        return cls(samples, list(set(all_labels)), weights)

    @staticmethod
    def _get_dataset_from_path(split: str):
        if split in ["validation", "test"]:
            with open(PREPARED_DIR / f"{split}.joblib", "rb") as fp:
                sentences = joblib.load(fp)
            return sentences
        with open(NOISE_DIR / f"noise_{split}.joblib", "rb") as fp:
            sentences = joblib.load(fp)
        return sentences


class UDPosTagggingPool(MultiLabelPool):
    """POS Tagging Pool"""

    @classmethod
    def prepare(cls, split: str):
        """Prepare Data Pool"""
        # get dataset from split
        train_dataset = UDPosTagggingPool._get_dataset_from_split(split)

        samples = []
        all_labels = []
        for data in train_dataset:
            token_texts = data["tokens"]
            token_texts_ = " ".join(token_texts)

            # check token to text
            flair_sentence = Sentence(token_texts_, use_tokenizer=False)
            assert len(flair_sentence.tokens) == len(token_texts)

            token_labels = data["ud_tags"]
            sample = Sample(input_text=token_texts_, oracle_label=token_labels)
            all_labels.extend(token_labels)
            samples.append(sample)
        weights = [1.0] * len(samples)
        return cls(samples, list(set(all_labels)), weights)

    @staticmethod
    def _get_dataset_from_split(split: str):
        if split == "train":
            return ud_pos_dataset(train=True)
        elif split == "val":
            return ud_pos_dataset(dev=True)
        elif split == "test":
            return ud_pos_dataset(test=True)


class CONLLNerTaggingPool(MultiLabelPool):
    """
    Note: Flair requires dataset files must be present under
    /root/.flair/datasets/conll03
    We can get the files from internet. For instance:
    https://github.com/ningshixian/NER-CONLL2003/tree/master/data
    """

    @classmethod
    def prepare(cls, split: str):
        """Prepare Data Pool"""
        # load the corpus
        corpus = datasets.CONLL_03()
        corpus_split = CONLLNerTaggingPool._get_dataset_from_corpus(corpus, split)

        samples = []
        all_labels = []
        for sentence in tqdm(corpus_split, desc="Preparing data pool"):
            token_texts = [token.text for token in sentence]
            token_texts_ = " ".join(token_texts)
            token_labels = [token.get_tag("ner").value for token in sentence]
            token_labels = [label.split("-")[1] if "-" in label else label for label in token_labels]  # simplify labels

            # check token to text
            flair_sentence = Sentence(token_texts_, use_tokenizer=False)
            assert len(flair_sentence.tokens) == len(token_texts)

            # sample
            sample = Sample(input_text=token_texts_, oracle_label=token_labels)
            samples.append(sample)
            all_labels.extend(token_labels)
        weights = [1.0] * len(samples)
        return cls(samples, list(set(all_labels)), weights)
