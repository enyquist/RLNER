# # standard libaries
# from copy import deepcopy
# from pathlib import Path
# from typing import List, Tuple

# # third party libraries
# import pandas as pd
# import spacy

# # custom libraries
# import utils
# from utils import Word

# ROOT_DIR = Path(__file__).resolve().parents[2]
# DATA_DIR = ROOT_DIR / "data"
# RAW_DIR = DATA_DIR / "raw"
# PREPARED_DIR = DATA_DIR / "prepared"

# nlp = spacy.load("en_core_web_lg")


# def bio_tagger(words: List[Word]) -> List[Word]:
#     max_multilabel_len = max([len(word.labels) for word in words])
#     words_out = []
#     prev_tag = ["O"] * max_multilabel_len
#     word_iter = 0
#     for word in words:
#         bio_tagged = []
#         labels = deepcopy(word.labels)

#         for idx, label in enumerate(labels):
#             if label == "O":
#                 bio_tagged.append(label)
#             elif label != "O" and prev_tag[idx] == "O":  # Begin NE
#                 bio_tagged.append("B-" + label)
#             elif prev_tag[idx] != "O" and prev_tag[idx] == label:  # Inside NE
#                 bio_tagged.append("I-" + label)
#             elif prev_tag[idx] != "O" and prev_tag[idx] != label:  # Adjacent NE
#                 bio_tagged.append("B-" + label)
#             prev_tag[idx] = label

#         # Reset secondary/tertiary labels if no extra labels
#         if len(labels) > 1:
#             word_iter += 1
#             if word_iter > 1:
#                 for idx in range(1, len(prev_tag)):
#                     prev_tag[idx] = "O"
#                 word_iter = 0

#         word.labels = bio_tagged

#         words_out.append(word)

#     return words


# def label_words(token_spans: List[Tuple[str, int]], entities: List[utils.Document]):
#     words = []

#     for (word, start_idx, end_idx) in token_spans:
#         word = Word(word, start_idx, end_idx)

#         for entity in entities:

#             START = entity["begin"]
#             END = entity["end"]
#             TYPE = entity["type"]

#             if START <= word.start_idx <= END and START <= word.end_idx <= END:
#                 if word.labels == ["O"]:
#                     word.labels = []
#                 word.labels.append(TYPE)

#         words.append(word)

#     return words


# def get_token_spans(text: str) -> List[Tuple[str, int]]:
#     doc = nlp(text)
#     token_spans = []
#     for token in doc:
#         token_span = doc[token.i : token.i + 1]
#         token_spans.append((token.text, token_span.start_char, token_span.end_char))

#     return token_spans


# def label_doc(document: utils.Document, doc_json: Path) -> pd.DataFrame:
#     ents_path = doc_json.parent / "entities.json"
#     ents = utils.read_json(ents_path)
#     docid = document["_id"]
#     doc_ents = [ent for ent in ents if ent["documentId"] == docid]

#     token_spans = get_token_spans(document["text"])
#     words = label_words(token_spans, doc_ents)
#     words = bio_tagger(words)
#     df = pd.DataFrame(words)
#     return df


# def main() -> None:
#     """Prepare data from raw format"""
#     doc_paths = RAW_DIR.glob("**/documents.json")
#     for doc_path in doc_paths:
#         docs = utils.read_json(doc_path)

#         dfs = [label_doc(document, doc_path) for document in docs]


# if __name__ == "__main__":
#     main()
