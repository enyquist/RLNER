# standard libaries
from typing import List


# All labels are implemented using sets
class HammingLoss:
    """Hamming Loss"""

    name = "hamming_loss"

    def __init__(self, total_labels: int):
        """Init"""
        self.total_labels = total_labels

    def __call__(self, true_labels: List[str], predicted_labels: List[str]) -> float:
        """Call"""
        incorrectly_predicted = [label for label in predicted_labels if label not in true_labels]
        missed_out_labels = [label for label in true_labels if label not in predicted_labels]
        n_incorrectly_predicted = len(incorrectly_predicted) + len(missed_out_labels)

        # hamming loss is computed when no labels are predicted (ie. missed out predictions)
        hamming_loss = (n_incorrectly_predicted) / self.total_labels
        return hamming_loss


class JaccardScore:
    """Jaccard Score"""

    name = "jaccard_score"

    def __call__(self, true_labels: List[str], predicted_labels: List[str]) -> float:
        """Call"""
        n_intersection = len(set(true_labels).intersection(set(predicted_labels)))
        n_union = len(set(true_labels).union(set(predicted_labels)))
        jaccard_score = n_intersection / n_union if n_union > 0 else 0.0
        return jaccard_score


class PrecisionScore:
    """Precision Score"""

    name = "precision_score"

    def __call__(self, true_labels: List[str], predicted_labels: List[str]) -> float:
        """Call"""
        n_correctly_predicted = len(set(true_labels).intersection(set(predicted_labels)))
        n_predicted_labels = len(predicted_labels)  # note repeated elements will be penalized
        precision = n_correctly_predicted / n_predicted_labels if n_predicted_labels > 0 else 0.0
        return precision


class RecallScore:
    """Recall Score"""

    name = "recall_score"

    def __call__(self, true_labels: List[str], predicted_labels: List[str]) -> float:
        """Call"""
        n_correctly_predicted = len(set(true_labels).intersection(set(predicted_labels)))
        n_true_labels = len(set(true_labels))
        recall = n_correctly_predicted / n_true_labels if n_true_labels > 0 else 0.0
        return recall


class F1Score:
    """F1 Score"""

    name = "f1_score"

    def __call__(self, true_labels: List[str], predicted_labels: List[str]) -> float:
        """Call"""
        precison = PrecisionScore()(true_labels, predicted_labels)
        recall = RecallScore()(true_labels, predicted_labels)
        if precison + recall > 0:
            f1_score = 2 * (precison * recall) / (precison + recall)
        else:
            f1_score = 0.0
        return f1_score


class F1ScoreCorpus:
    """F1 score on a corpus"""

    name = "f1_score"

    def __call__(self, true_labels: List[List[str]], predicted_labels: List[List[str]]) -> float:
        """Call"""
        total_score = 0.0
        for true, predicted in zip(true_labels, predicted_labels):

            # remove terminate
            if "terminate" in predicted:
                predicted.remove("terminate")

            score = F1Score()(true, predicted)
            total_score += score
        score = {"average_f1_score": total_score / len(true_labels)}
        return score
