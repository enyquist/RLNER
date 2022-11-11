# standard libaries
import io
from pathlib import Path
from typing import Dict, Tuple

# third party libraries
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from crf import CRF

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"


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
    model2 = models.Model(inputs=[inputs], outputs=[pred, sh_fw, sc_fw, sh_bw, sc_bw, embeddings])

    model1.compile(optimizer="adam", loss=crf.loss, metrics=[crf.accuracy])
    model1.summary()

    return model1, model2
