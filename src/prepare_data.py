# standard libaries
import logging
from ast import literal_eval
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

# third party libraries
import pandas as pd
import spacy
from tqdm import tqdm

# rlner libraries
# custom libraries
from utils import Word

logger = logging.getLogger()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPARED_DIR = DATA_DIR / "prepared"

Re3dDict = Dict[str, Union[str, int]]
TokenSpan = List[Tuple[int, str, int, int]]

nlp = spacy.load("en_core_web_lg")


def match_doc_id(doc: Re3dDict, line: Re3dDict) -> bool:
    """Match entities to document by document ID

    Args:
        doc (Re3dDict): Re3d Document dictionary
        line (Re3dDict): Re3d Entity dictionary

    Returns:
        bool: True if document id matches between arguments
    """
    return doc["_id"] == line["documentId"]


def merge_dfs(dfs: List[List[pd.DataFrame]]) -> pd.DataFrame:
    """Merge nested lists of dataframes into one dataframe

    Args:
        dfs (List[List[pd.DataFrame]]): Nested dataframes

    Returns:
        pd.DataFrame: Merged dataframe
    """
    flat_dfs = [item for sublist in dfs for item in sublist]
    base_df = flat_dfs.pop()

    while flat_dfs:
        df = flat_dfs.pop()
        max_sentence_num = max(base_df["sentence_num"])
        df["sentence_num"] = df["sentence_num"].apply(lambda x: x + max_sentence_num + 1)
        base_df = pd.concat([base_df, df], axis=0)

    return base_df


def bio_tagger(words: List[Word]) -> List[Word]:
    """Format Re3d Entity tags into BIO schema

    Args:
        words (List[Word]): List of Words with Re3d Schema

    Returns:
        List[Word]: Words with BIO Schema
    """
    max_multilabel_len = max([len(word.tags) for word in words])
    words_out = []
    prev_tag = ["O"] * max_multilabel_len
    word_iter = 0
    for _, word in enumerate(words):
        bio_tagged = []
        _, labels = deepcopy(word.word), deepcopy(word.tags)

        for idx, label in enumerate(labels):
            if label == "O":
                bio_tagged.append(label)
            elif label != "O" and prev_tag[idx] == "O":  # Begin NE
                bio_tagged.append("B-" + label)
            elif prev_tag[idx] != "O" and prev_tag[idx] == label:  # Inside NE
                bio_tagged.append("I-" + label)
            elif prev_tag[idx] != "O" and prev_tag[idx] != label:  # Adjacent NE
                bio_tagged.append("B-" + label)
            prev_tag[idx] = label

        # Reset secondary/tertiary labels if no extra labels
        if len(labels) > 1:
            word_iter += 1
            if word_iter > 1:
                for idx in range(1, len(prev_tag)):
                    prev_tag[idx] = "O"
                word_iter = 0

        word.tags = bio_tagged

        words_out.append(word)

    return words


def validate_entity(entity: Re3dDict, token_spans: TokenSpan) -> bool:
    """Check that an Re3d Entity dict's entity span matches a word boundaries.

    Some entities in Re3d start inside words, this ensures only entities that align with
    word boundaries are valid.

    Args:
        entity (Re3dDict): Re3d Entity Dictionary
        token_spans (TokenSpan): Token spans

    Returns:
        bool: Valid Entity
    """
    if entity["begin"] not in [tup[2] for tup in token_spans]:
        return False
    return True


def label_words(token_spans: TokenSpan, entities: List[Re3dDict]) -> List[Word]:
    """Label individual words with Re3d entity tags

    Args:
        token_spans (TokenSpan): Token spans
        entities (List[Re3dDict]): List of entities

    Returns:
        List[Word]: List of Re3d labeled words
    """
    words = []

    for sentence_num, text, start_idx, end_idx in token_spans:
        word = Word(sentence_num, text, start_idx, end_idx)

        for entity in entities:
            if not validate_entity(entity, token_spans):
                continue

            START = entity["begin"]
            END = entity["end"]
            TYPE = entity["type"]

            if START <= word.start_idx <= END and START <= word.end_idx <= END:
                if word.tags == ["O"]:
                    word.tags = []
                word.tags.append(TYPE)

        words.append(word)

    return words


def get_token_spans(text: str) -> TokenSpan:
    """Generate Token Spans given text

    Args:
        text (str): raw text

    Returns:
        TokenSpan: List of (sentence_idx, word, index_start, index_end)
    """

    doc = nlp(text)
    token_spans = []
    for sentence_idx, sent in enumerate(doc.sents):
        for token in sent:
            token_span = doc[token.i : token.i + 1]
            token_spans.append((sentence_idx, token.text, token_span.start_char, token_span.end_char))

    # Remove problematic tokens
    blank_tokens = []
    for idx, span in enumerate(token_spans):
        if any(ext in span[1] for ext in ["\n", " ", "\xa0"]):
            blank_tokens.append(idx)

    for idx in reversed(blank_tokens):
        token_spans.pop(idx)

    return token_spans


def preprocess_docs(doc_path: Path) -> List[pd.DataFrame]:
    """ETL a given documents.json into a dataframe with sentence number,
    word, start index, end index, and BIO tags

    Args:
        doc_path (Path): Path to */documents.json

    Returns:
        List[pd.DataFrame]: Dataframe for each document in doc_path
    """
    # load docs
    with open(doc_path, "r") as fp:
        docs = [literal_eval(line) for line in fp]

    entities_path = doc_path.parent / "entities.json"

    output = []

    for doc in docs:
        # Load entities for doc
        with open(entities_path, "r") as fp:
            entities = [literal_eval(line) for line in fp if match_doc_id(doc, literal_eval(line))]

        # ID sentence number, token, start/end idx
        token_spans = get_token_spans(doc["text"])

        # Add tags
        labeled_words = label_words(token_spans, entities)

        # Format tags as BIO
        tagged_words = bio_tagger(labeled_words)

        # Create dataframe
        doc_df = pd.DataFrame([asdict(word) for word in tagged_words])

        output.append(doc_df)

    return output


def get_pos_tag(text: str) -> str:
    """Get POS tag for token

    Args:
        text (str): text

    Returns:
        str: POS tag
    """
    doc = nlp(text)
    return [tok.pos_ for tok in doc][0]


def main() -> None:
    """Transform raw Re3d Data into a master csv"""

    docs_list = list(RAW_DIR.glob("**/documents.json"))

    # Process docs
    dfs = [preprocess_docs(doc_path) for doc_path in tqdm(docs_list, desc="Formatting Doc Paths")]

    # Merge docs
    master_df = merge_dfs(dfs)
    master_df.reset_index(drop=True, inplace=True)

    # Split single labels out instead of multi-label and get POS
    master_df["single_tag"] = master_df["tags"].apply(lambda x: x[0])
    master_df["POS"] = master_df["word"].apply(get_pos_tag)

    # Save master df
    master_df.to_csv(PREPARED_DIR / "master.csv", index=False)

    logger.info("Success")


if __name__ == "__main__":
    main()
