# standard libaries
import io
from pathlib import Path
from typing import Dict

# third party libraries
import numpy as np
from crf import CRF
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, TimeDistributed
from tensorflow.keras.models import Sequential

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"


def create_model(
    vocab_size: int, max_length: int, embedding_dim: int, word_index: Dict[str, int], tag_index: Dict[str, int]
) -> Sequential:
    """Create Bi-LSTM CRF model in tensorflow

    Args:
        vocab_size (int): Size of vocabulary
        max_length (int): Max sequence length
        embedding_dim (int): Size of embedding. Make sure to match size of GloVe embedding.
        word_index (Dict[str, int]): Index mapping words to ints
        tag_index (Dict[str, int]): Index mapping tokens to ints

    Returns:
        Sequential: Compiled Sequential Model
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

    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            weights=[embeddings_matrix],
            mask_zero=True,
        )
    )

    model.add(Bidirectional(LSTM(units=embedding_dim, return_sequences=True, recurrent_dropout=0.01)))

    model.add(TimeDistributed(Dense(len(tag_index))))

    crf = CRF(len(tag_index), sparse_target=True)
    model.add(crf)

    model.compile(optimizer="adam", loss=crf.loss, metrics=["crf.accuracy"])
    model.summary()

    return model
