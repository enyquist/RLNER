# third party libraries
import flair
import torch
from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings

flair.device = torch.device("cpu")


class DocEmbeddings:
    """Doc Embeddings"""

    __instance = None

    @staticmethod
    def getInstance():
        """Static access method."""
        if DocEmbeddings.__instance is None:
            DocEmbeddings()
        return DocEmbeddings.__instance

    def __init__(self):
        """Virtually private constructor."""
        if DocEmbeddings.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            doc_embeddings = DocumentPoolEmbeddings([WordEmbeddings("glove")])
            DocEmbeddings.__instance = doc_embeddings
