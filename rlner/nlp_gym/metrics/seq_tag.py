# standard libaries
from typing import List

# third party libraries
from sklearn.metrics import f1_score, precision_score, recall_score


class EntityScores:
    """Score on entities"""

    def __init__(self, average: str = "micro"):
        """Init"""
        self.average = average

    def __call__(self, true_labels: List[str], predicted_labels: List[str]):
        """Call"""
        metrics_dict = {
            "precision": precision_score(true_labels, predicted_labels, average=self.average, zero_division=0)
            if len(true_labels) > 0
            else 0.0,
            "recall": recall_score(true_labels, predicted_labels, average=self.average, zero_division=0)
            if len(true_labels) > 0
            else 0.0,
            "f1": f1_score(true_labels, predicted_labels, average=self.average, zero_division=0)
            if len(true_labels) > 0
            else 0.0,
        }
        return metrics_dict


class EntityScoresCorpus:
    """Score across a corpus"""

    def __init__(self, average: str = "micro"):
        """Init"""
        self.average = average

    def __call__(self, true_labels: List[List[str]], predicted_labels: List[List[str]]):
        """Call"""
        all_true = []
        all_predicted = []
        for true, predicted in zip(true_labels, predicted_labels):
            all_true.extend(true)
            all_predicted.extend(predicted)

        metrics_dict = {
            "precision": precision_score(all_true, all_predicted, average=self.average)
            if len(true_labels) > 0
            else 0.0,
            "recall": recall_score(all_true, all_predicted, average=self.average) if len(true_labels) > 0 else 0.0,
            "f1": f1_score(all_true, all_predicted, average=self.average) if len(true_labels) > 0 else 0.0,
        }
        return metrics_dict


if __name__ == "__main__":
    metric_fn = EntityScores(average="macro")
    sent_true_labels = ["PER", "LOC", "0"]
    sent_predicted_labels = ["0", "0", "0"]
    print(metric_fn(sent_true_labels, sent_predicted_labels))
