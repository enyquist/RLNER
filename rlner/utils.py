# standard libaries
import io
from pathlib import Path
from typing import Dict, Tuple

# third party libraries
import joblib
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from tensorflow.keras.preprocessing.sequence import pad_sequences

# rlner libraries
from rlner.crf import CRF

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"
PREPARED_DIR = DATA_DIR / "prepared"
NOISE_DIR = PREPARED_DIR / "noise"


def create_model(
    vocab_size: int, max_length: int, embedding_dim: int, word_index: Dict[str, int], tag_index: Dict[str, int]
) -> Tuple[models.Model]:
    """Create Bi-LSTM CRF model in tensorflow.

    Model1 is the trainable model. Model2 is for predictions and returns:
    [predicted labels, LSTM hidden state (Forward and backward), LSTM cell state (forward and backward), embeddings]

    This is leveraged to build the REINFORCE states.

    Adapted from:
    https://github.com/ngoquanghuy99/POS-Tagging-BiLSTM-CRF

    Args:
        vocab_size (int): Size of vocabulary
        max_length (int): Max sequence length
        embedding_dim (int): Size of embedding. Make sure to match size of GloVe embedding.
        word_index (Dict[str, int]): Index mapping words to ints
        tag_index (Dict[str, int]): Index mapping tokens to ints

    Returns:
        Tuple[Model]: Compiled Model and Non-compiled Model
        with exposed LSTM and embedding layers
    """

    embeddings_index = {}
    with io.open(EMBEDDING_DIR / "glove.6B.100d.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            curr_word = values[0]
            coefs = np.asarray(values[1:], dtype="float64")
            embeddings_index[curr_word] = coefs
        embeddings_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in word_index.items():
            if i > vocab_size:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embeddings_matrix[i] = embedding_vector

    inputs = layers.Input(shape=(max_length,))

    embeddings = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        weights=[embeddings_matrix],
        mask_zero=True,
    )(inputs)

    lstm_out, sh_fw, sc_fw, sh_bw, sc_bw = layers.Bidirectional(
        layers.LSTM(units=embedding_dim, return_sequences=True, return_state=True, recurrent_dropout=0.01)
    )(embeddings)

    time_dist = layers.TimeDistributed(layers.Dense(len(tag_index)))(lstm_out)

    crf = CRF(len(tag_index), sparse_target=False)
    pred = crf(time_dist)

    model1 = models.Model(inputs=[inputs], outputs=[pred])
    model2 = models.Model(inputs=[inputs], outputs=[pred, lstm_out, sh_fw, sc_fw, sh_bw, sc_bw, embeddings])

    model1.compile(optimizer="adam", loss=crf.loss, metrics=[crf.accuracy])
    model1.summary()

    return model1, model2


class SentenceGetter:
    """Iterator to get a sentence sequence and its BIO tags"""

    def __init__(self, data):
        """Init"""
        self.n_sent = 1
        self.data = data
        self.empty = False
        self.grouped = self.data.groupby("sentence_num").apply(self._agg_func)
        self.sentences = [s for s in self.grouped]

    @staticmethod
    def _agg_func(s):
        """Aggregate word, pos, tag from grouped sentence"""
        return [
            (w, pos, t)
            for w, pos, t in zip(s["word"].values.tolist(), s["POS"].values.tolist(), s["single_tag"].values.tolist())
        ]

    def get_next(self):
        """Get next sentence"""
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except KeyError:
            return None


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


def _output_to_tag_sequence(pred):
    return np.argmax(pred, axis=1)


def _encode(test_sentence, words2index):
    return np.array([words2index.get(word[0], words2index.get("PADword")) for word in test_sentence])


def _decode_output(sequence, index2tags):
    return [index2tags[idx] for idx in sequence]


def _make_preds(test_sentences, max_len, words2index, index2tags, model):
    sequences = [_encode(sent, words2index) for sent in test_sentences]
    sequences = pad_sequences(maxlen=max_len, sequences=sequences, padding="post", value=words2index["PADword"])
    preds, *rest = model.predict(sequences)
    preds = [_output_to_tag_sequence(pred) for pred in preds]
    decoded_preds = [_decode_output(sequence, index2tags) for sequence in preds]
    for sent, decoded_pred in zip(test_sentences, decoded_preds):
        for idx, (token, pred) in enumerate(zip(sent, decoded_pred)):
            sent[idx] = token + (pred,)

    return test_sentences


def _pred_lstm(split: float, model: tf.keras.models.Model, train_sentences, test_sentences):
    max_len = 100

    words = _get_words(train_sentences)
    tags = _get_tags(train_sentences)
    words2index = _to_idx(words)
    tags2index = _to_idx(tags)
    index2tags = {idx: tag for tag, idx in tags2index.items()}

    test_sentences = _make_preds(test_sentences, max_len, words2index, index2tags, model)

    with open(MODEL_DIR / f"agents/agent_{split}/model_preds.txt", "w") as fp:
        for sentence in test_sentences:
            for token, pos, truth, pred in sentence:
                fp.write(f"{token} {pos} {truth} {pred}\n")
            fp.write("\n")
