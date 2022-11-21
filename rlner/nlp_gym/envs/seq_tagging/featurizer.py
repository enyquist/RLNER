# standard libaries
from typing import List, Union

# third party libraries
import spacy
import tensorflow as tf
import torch
from flair.data import Sentence
from flair.embeddings import BytePairEmbeddings, Embeddings, StackedEmbeddings, WordEmbeddings
from tensorflow.keras.preprocessing.sequence import pad_sequences

# rlner libraries
from rlner.nlp_gym.envs.common.action_space import ActionSpace
from rlner.nlp_gym.envs.seq_tagging.observation import Observation, ObservationFeaturizer

nlp = spacy.load("en_core_web_lg")


class EmbeddingRegistry:
    """Embedding Regirstry"""

    _registry_mapping = {
        "byte_pair": {"cls": [BytePairEmbeddings], "params": ["en"]},
        "fasttext": {"cls": [WordEmbeddings], "params": ["en-crawl"]},
        "fasttext_de": {"cls": [WordEmbeddings], "params": ["de-crawl"]},
    }

    def get_embedding(embedding_type: str) -> List[Embeddings]:
        """Get embedding"""
        cls_ = EmbeddingRegistry._registry_mapping[embedding_type]["cls"]
        params_ = EmbeddingRegistry._registry_mapping[embedding_type]["params"]
        embeddings = [embedding_cls(embedding_param) for embedding_cls, embedding_param in zip(cls_, params_)]
        return embeddings


class DefaultFeaturizerForSeqTagging(ObservationFeaturizer):
    """Featurize observations"""

    def __init__(self, action_space: ActionSpace, embedding_type: str = "fasttext", device: str = "cpu"):
        """Init"""
        self.device = device
        self._setup_device()
        embeddings = EmbeddingRegistry.get_embedding(embedding_type)
        self.doc_embeddings = StackedEmbeddings(embeddings).to(torch.device(self.device))
        self.action_space = action_space
        self._current_token_embeddings: List[torch.tensor] = None

    def _setup_device(self):
        # third party libraries
        import flair
        import torch

        flair.device = torch.device(self.device)

    def init_on_reset(self, input_text: Union[List[str], str]):
        """Init on reset"""
        sent = Sentence(input_text)
        self.doc_embeddings.embed(sent)
        self._current_token_embeddings = [token.embedding.cpu().detach() for token in sent]
        sent.clear_embeddings()

    def featurize(self, observation: Observation) -> torch.Tensor:
        """Featurize observation"""
        input_vector = self._featurize_input(observation.get_current_index())
        context_vector = self._featurize_context(observation.get_current_action_history())
        concatenated = torch.cat((input_vector, context_vector), dim=0)
        return concatenated

    def get_observation_dim(self) -> int:
        """Utility function"""
        return self._get_input_dim() + self._get_context_dim()

    def _featurize_input(self, input_index: int) -> torch.Tensor:
        input_features = self._current_token_embeddings[input_index]
        return input_features

    def _featurize_context(self, context: List[str]) -> torch.Tensor:
        # consider only last action
        context_vector = torch.zeros(self.action_space.size())
        context_ = [context[-1]] if len(context) > 0 else []
        action_indices = [self.action_space.action_to_ix(action) for action in context_]
        context_vector[action_indices] = 1.0
        return context_vector

    def _get_input_dim(self):
        sent = Sentence("A random text to get the embedding dimension")
        self.doc_embeddings.embed(sent)
        dim = sent[0].embedding.shape[0]
        sent.clear_embeddings()
        return dim

    def _get_context_dim(self):
        return self.action_space.size()


class BiLSTMFeaturizerForSeqTagging(DefaultFeaturizerForSeqTagging):
    """Featurize with BiLSTM features"""

    def __init__(
        self,
        action_space: ActionSpace,
        word2index,
        model,
        max_len,
        embedding_type: str = "fasttext",
        device: str = "cpu",
    ):
        """Init"""
        super().__init__(action_space=action_space, embedding_type=embedding_type, device=device)
        self.word2index = word2index
        self.model = model
        self.max_len = max_len

    def init_on_reset(self, input_text: Union[List[str], str]):
        """Init on reset"""
        doc = nlp(input_text)
        sent = [token.text for token in doc]
        # sent = Sentence(input_text)
        X = [[self.word2index.get(w, self.word2index.get("PADword")) for w in sent]]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.word2index["PADword"])
        x_tensor = tf.convert_to_tensor(X)
        preds = self.model.predict(x_tensor)
        self._current_token_embeddings = preds[-1]
        self._lstm = preds[1]
        self._sh_fw = preds[2]
        self._sc_fw = preds[3]
        self._sh_bw = preds[4]
        self._sc_bw = preds[5]
        # sent.clear_embeddings()

    def _get_input_dim(self):
        return 100

    def featurize(self, observation: Observation) -> torch.Tensor:
        """Featurize observation"""
        input_vector = self._featurize_input(observation.get_current_index())
        context_vector = self._featurize_context(observation.get_current_index())
        concatenated = torch.cat((context_vector, input_vector), dim=0)
        return concatenated

    def _featurize_input(self, input_index: int) -> torch.Tensor:
        input_features = self._current_token_embeddings[0][input_index]
        return torch.from_numpy(input_features)

    def _featurize_context(self, input_index: int) -> torch.Tensor:
        if input_index > 0:
            context = torch.from_numpy(self._lstm[0][input_index])
            # sh_fw = torch.from_numpy(self._sh_fw[0])
            # sc_fw = torch.from_numpy(self._sc_fw[0])
            # sh_bw = torch.from_numpy(self._sh_bw[0])
            # sc_bw = torch.from_numpy(self._sc_bw[0])

        else:
            # sh_fw = torch.zeros(self._sh_fw[0].shape)
            # sc_fw = torch.zeros(self._sc_fw[0].shape)
            # sh_bw = torch.zeros(self._sh_bw[0].shape)
            # sc_bw = torch.zeros(self._sc_bw[0].shape)
            context = torch.zeros(self._lstm.shape[-1])

        # concatenated = torch.cat((sc_fw, sc_bw, sh_fw, sh_bw), dim=0)
        return context

    def _get_context_dim(self):
        return 200
