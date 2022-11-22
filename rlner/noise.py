# standard libaries
import random
from abc import ABC, abstractclassmethod
from copy import deepcopy
from typing import List, Tuple

Sentence = List[Tuple[str]]
Sentences = List[Sentence]


LABELS = {
    "Location",
    "Organisation",
    "Person",
    "Quantity",
    "Temporal",
}


def add_noise(sentences: Sentences, noise_percentage: float = 0.0) -> Sentences:
    """Apply noise to Sentences

    Args:
        sentences (noise.Sentences): Training Sentences
        noise_percentage (float, optional): Amount of noise to apply to sentences
        as a percentage. Defaults to 0.0.

    Raises:
        ValueError: Raise if noise_percentage not in [0, 1]

    Returns:
        noise.Sentences: Noisy Sentences
    """
    if not 0 <= noise_percentage <= 1.0:
        raise ValueError(f"noise_percentage must be between 0.0 and 1.0: found {noise_percentage}")

    noisy_sentences = []

    num_entities = get_total_entities(sentences)

    total_noisy_entities = int(num_entities * noise_percentage)
    total_sentences = len(sentences)
    partition_2 = int(total_sentences * 0.3)
    partition_3 = partition_2 * 2

    removed_sentences = sentences[:partition_2]
    altered_sentences = sentences[partition_2:partition_3]
    ambigious_sentences = sentences[partition_3:]

    removed_sentences = RemovedNoiseGenerator.apply_noise(
        removed_sentences, num_entities_to_edit=total_noisy_entities // 3
    )
    altered_sentences = AlteredNoiseGenerator.apply_noise(
        altered_sentences, num_entities_to_edit=total_noisy_entities // 3
    )
    ambigious_sentences = AmbigiousNoiseGenerator.apply_noise(
        ambigious_sentences, num_entities_to_edit=total_noisy_entities // 3
    )

    noisy_sentences.extend(removed_sentences)
    noisy_sentences.extend(altered_sentences)
    noisy_sentences.extend(ambigious_sentences)

    return noisy_sentences


def get_total_entities(sentences: Sentences) -> int:
    """Get the totol number of entities in a set of Sentences"""
    entities = [[ent for ent in sent if ent[-1] != "O"] for sent in sentences]
    num_entities = [len([ent for ent in sent if ent[-1].startswith("B")]) for sent in entities]
    return sum(num_entities)


class NoiseGenerator(ABC):
    """Template for generating and applying noise to sentence tags"""

    @classmethod
    def apply_noise(cls, sentences: Sentences, num_entities_to_edit: int = 0) -> Sentences:
        """Apply Noise to sentences. Noise function depends on class.
        Child classes must implement noise function

        Args:
            sentences (Sentences): Sentences with BIO Tags
            num_entities_to_edit (int, optional): Number of entities to edit. Defaults to 0.

        Returns:
            Sentences: Noisy Sentences
        """
        cls.num_entities_to_edit = num_entities_to_edit
        cls.num_noisy_entities = 0
        output_sentences = deepcopy(sentences)
        seen_indices = []
        current_idx_options = list(range(len(sentences)))

        while cls.num_noisy_entities < cls.num_entities_to_edit and current_idx_options:
            # Get random sentence
            done = False

            while not done:
                rand_idx = random.sample(current_idx_options, 1)[0]
                if rand_idx not in seen_indices:
                    done = True
                    seen_indices.append(rand_idx)
                    current_idx_options.remove(rand_idx)

            random_sent = output_sentences.pop(rand_idx)

            # Get entities in sentence
            entities = [(idx, ent) for idx, ent in enumerate(random_sent) if ent[-1] != "O"]

            # Make entity blocks
            grouped_entities = cls._group_entities(entities)

            # Apply Noise
            noisy_entities = cls.noise(grouped_entities, random_sent)

            # Update sentence with noisy entities
            for idx, entity in noisy_entities:
                random_sent[idx] = entity

            # Add updated random sentence to output
            output_sentences.append(random_sent)

        # Sanity checks
        assert len(output_sentences) == len(sentences)
        assert cls.num_noisy_entities <= cls.num_entities_to_edit

        return output_sentences

    @abstractclassmethod
    def noise(cls, *args) -> List[Tuple[str]]:
        """Apply noise to sentences"""
        raise NotImplementedError

    @classmethod
    def _group_entities(cls, entities: List[Tuple[str]], step: int = 1) -> List[List[Tuple[str]]]:
        """Group entities by index.
        Entities come as individual BIO tags, method groups them into complete entities.

        Args:
            entities (List[Tuple[str]]): Entities
            step (int, optional): How many tokens to consider part of the entity. Defaults to 1.

        Returns:
            List[List[Tuple[str]]]: Grouped Entities
        """
        output = []
        for ent in entities:
            if output and output[-1] and ent[0] - step == output[-1][-1][0]:
                output[-1].append(ent)
            else:
                output.append([ent])
        return output


class RemovedNoiseGenerator(NoiseGenerator):
    """Remove entity tags to simulate missing entities"""

    @classmethod
    def noise(cls, grouped_entities: List[List[Tuple[str]]], random_sent: Sentence = None) -> List[Tuple[str]]:
        """Apply Noise to entities.

        Simulates missing entities by changing the BIO tags to "O"

        Example::

        United -> B-GPE     Apply Noise     United  -> O
        States -> I-GPE                     States  -> O

        Args:
            grouped_entities (List[List[Tuple[str]]]): Complete entities with BIO tags
            random_sent (Sentence, optional): Random sentence noise is applied to. Complies with API. Defaults to None.

        Returns:
            List[Tuple[str]]: Noisy entities
        """
        noisy_entities = []
        for entity_block in grouped_entities:
            cls.num_noisy_entities += 1
            for idx, entity in entity_block:
                entity = list(entity)
                entity[-1] = "O"
                noisy_entities.append((idx, tuple(entity)))

            if cls.num_noisy_entities == cls.num_entities_to_edit:
                # stop editing entities in grouped_entities
                break

        return noisy_entities


class AlteredNoiseGenerator(NoiseGenerator):
    """Remove entity tags to simulate mislabled entities"""

    @classmethod
    def noise(cls, grouped_entities: List[List[Tuple[str]]], random_sent: Sentence = None) -> List[Tuple[str]]:
        """Apply Noise to entities.

        Simulates mislabeled entities by changing the labels to another class while preserving BIO schema.

        Example::

        United -> B-GPE     Apply Noise     United  -> B-ORG
        States -> I-GPE                     States  -> I-ORG


        Args:
            grouped_entities (List[List[Tuple[str]]]): Complete entities with BIO tags
            random_sent (Sentence, optional): Random sentence noise is applied to. Complies with API. Defaults to None.

        Returns:
            List[Tuple[str]]: Noisy entities
        """
        noisy_entities = []
        for entity_block in grouped_entities:
            cls.num_noisy_entities += 1
            for idx, entity in entity_block:
                temp_entity_labels = deepcopy(LABELS)
                entity = list(entity)
                bio_tag, label = entity[-1].split("-")
                temp_entity_labels.remove(label)
                new_label = random.choice(tuple(temp_entity_labels))
                entity[-1] = bio_tag + "-" + new_label
                noisy_entities.append((idx, tuple(entity)))

            if cls.num_noisy_entities == cls.num_entities_to_edit:
                # stop editing entities in grouped_entities
                break

        return noisy_entities


class AmbigiousNoiseGenerator(NoiseGenerator):
    """
    Remove entity tags to simulate ambigous entities.
    Entities with an extra B/I tag
    """

    @classmethod
    def noise(cls, grouped_entities: List[List[Tuple[str]]], random_sent: Sentence = None) -> List[Tuple[str]]:
        """Apply Noise to entities.

        Simulates ambigious entities by editing the entity span. Mimics poor labeling by humans.
        Only edits the start or end span for a given entity

        Example:: Edit start span

        the    -> O             Apply Noise     the     -> B-GPE
        United -> B-GPE                         United  -> I-GPE
        States -> I-GPE                         States  -> I-GPE

        Example:: Edit end span

        Ambassador -> B-PER     Apply Noise     Ambassador  -> B-PER
        to         -> O                         to          -> I-PER


        Args:
            grouped_entities (List[List[Tuple[str]]]): Complete entities with BIO tags
            random_sent (Sentence, optional): Random sentence noise is applied to. Complies with API. Defaults to None.

        Returns:
            List[Tuple[str]]: Noisy entities
        """
        noisy_entities = []
        for entity_block in grouped_entities:
            first_idx = min(entity_block)
            last_idx = max(entity_block)

            # Try to edit the first tag
            if random.uniform(0, 1) < 0.5:
                # Check if there is a tag prior to the tag in question
                if first_idx[0] > 0:
                    cls.num_noisy_entities += 1
                    # Parse and Update tags
                    entity = list(first_idx[1])
                    prev_token = list(random_sent[first_idx[0] - 1])
                    _, label = entity[-1].split("-")
                    entity[-1] = "I-" + label
                    prev_token[-1] = "B-" + label
                    noisy_entities.append((first_idx[0], tuple(entity)))
                    noisy_entities.append((first_idx[0] - 1, tuple(prev_token)))

            # Try to edit the last tag
            else:
                # Check if there is a tag after the tag in question
                if last_idx[0] + 1 < len(random_sent):
                    cls.num_noisy_entities += 1
                    # Parse and Update tags
                    entity = list(last_idx[1])
                    next_token = list(random_sent[last_idx[0] + 1])
                    _, label = entity[-1].split("-")
                    next_token[-1] = "I-" + label
                    noisy_entities.append((last_idx[0] + 1, tuple(next_token)))

            if cls.num_noisy_entities == cls.num_entities_to_edit:
                # stop editing entities in grouped_entities
                break

        return noisy_entities
