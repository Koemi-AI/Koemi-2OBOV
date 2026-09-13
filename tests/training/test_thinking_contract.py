from __future__ import annotations

import logging
import math
import unittest

import torch

from koemi.configuration.settings import ModelSettings, TrainingSettings
from koemi.data.contracts import DatasetRecord
from koemi.data.serialization import serialize_record
from koemi.model.network import KoemiModel
from koemi.training.dataset import CausalByteDataset, create_training_loader
from koemi.training.objective import calculate_training_objective, token_cross_entropy
from koemi.training.trainer import MetricAccumulator, Trainer


def build_model() -> KoemiModel:
    return KoemiModel(ModelSettings(embedding_size=16, memory_features=4, local_memory_size=4))


class CountingTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__(logging.getLogger("koemi-counting-thinking-contract"))
        self.evaluation_calls = 0

    def evaluate(self, *args, **kwargs):
        self.evaluation_calls += 1
        return super().evaluate(*args, **kwargs)


class ThinkingContractTests(unittest.TestCase):
    def test_objective_reports_answer_loss_separately_from_thinking_loss(self) -> None:
        model = build_model()
        output = model(torch.tensor([[65, 66, 67, 68]], dtype=torch.long))
        target_ids = torch.tensor([[66, 67, 68, 69]], dtype=torch.long)
        thinking_mask = torch.tensor([[False, True, True, False]])

        objective = calculate_training_objective(output, target_ids, thinking_mask, 3.0)
        token_loss = token_cross_entropy(output.logits, target_ids)
        answer_positions = ~thinking_mask
        expected_answer_loss = (token_loss * answer_positions).sum() / int(answer_positions.sum())

        self.assertTrue(torch.allclose(objective.answer_loss, expected_answer_loss))
        self.assertNotAlmostEqual(
            float(objective.answer_loss.detach()), float(objective.thinking_loss.detach()), places=6
        )

    def test_weighted_objective_matches_the_token_level_reference(self) -> None:
        model = build_model()
        output = model(torch.tensor([[65, 66, 67, 68]], dtype=torch.long))
        target_ids = torch.tensor([[66, -100, 68, 69]], dtype=torch.long)
        thinking_mask = torch.tensor([[False, True, True, False]])
        thinking_loss_weight = 2.5

        objective = calculate_training_objective(
            output, target_ids, thinking_mask, thinking_loss_weight
        )
        token_loss = token_cross_entropy(output.logits, target_ids)
        supervised_positions = target_ids >= 0
        thinking_positions = supervised_positions & thinking_mask
        answer_positions = supervised_positions & ~thinking_mask
        thinking_loss_sum = (token_loss * thinking_positions).sum()
        answer_loss_sum = (token_loss * answer_positions).sum()
        expected_total_loss = (answer_loss_sum + thinking_loss_weight * thinking_loss_sum) / (
            int(answer_positions.sum()) + thinking_loss_weight * int(thinking_positions.sum())
        )

        self.assertTrue(torch.allclose(objective.total_loss, expected_total_loss))

    def test_metric_accumulator_weights_thinking_loss_by_thinking_tokens(self) -> None:
        model = build_model()
        answer_output = model(torch.tensor([[65, 66]], dtype=torch.long))
        answer_targets = torch.tensor([[66, 67]], dtype=torch.long)
        answer_mask = torch.zeros_like(answer_targets, dtype=torch.bool)
        answer_objective = calculate_training_objective(answer_output, answer_targets, answer_mask, 2.0)
        thinking_output = model(torch.tensor([[70, 71, 72, 73]], dtype=torch.long))
        thinking_targets = torch.tensor([[71, 72, 73, 74]], dtype=torch.long)
        thinking_mask = torch.tensor([[False, True, True, True]])
        thinking_objective = calculate_training_objective(thinking_output, thinking_targets, thinking_mask, 2.0)

        accumulator = MetricAccumulator()
        accumulator.add(answer_output, answer_objective, supervised_token_count=2, thinking_token_count=0)
        accumulator.add(thinking_output, thinking_objective, supervised_token_count=4, thinking_token_count=3)

        self.assertAlmostEqual(
            float(thinking_objective.thinking_loss.detach()), accumulator.mean_thinking_loss, places=6
        )

    def test_dataset_discards_chunks_without_a_supervised_target(self) -> None:
        record = DatasetRecord("long-prompt", "Context " * 40, "Reason", "Answer", {})
        sequence_length = 16
        dataset = CausalByteDataset((record,), sequence_length)
        serialized_record = serialize_record(record)
        unfiltered_chunk_count = math.ceil((len(serialized_record.token_bytes) - 1) / sequence_length)

        self.assertLess(len(dataset), unfiltered_chunk_count)
        self.assertTrue(
            all(any(target_id >= 0 for target_id in chunk.target_ids) for chunk in dataset.chunks)
        )

    def test_training_reports_answer_bpb(self) -> None:
        records = (DatasetRecord("reasoned", "Question", "Reason", "Answer", {}),)
        dataset = CausalByteDataset(records, sequence_length=64)
        result = Trainer(logging.getLogger("koemi-thinking-contract")).train(
            build_model(),
            create_training_loader(dataset, batch_size=1, shuffle=False),
            TrainingSettings(sequence_length=64, batch_size=1, epochs=1, device="cpu", thinking_loss_weight=2.0),
        )

        self.assertIsNotNone(result.mean_answer_loss)
        self.assertIsNotNone(result.mean_answer_bpb)
        self.assertAlmostEqual(result.mean_answer_loss / math.log(2.0), result.mean_answer_bpb, places=6)

    def test_training_reuses_the_last_validation_result(self) -> None:
        records = (DatasetRecord("reasoned", "Question", "Reason", "Answer", {}),)
        dataset = CausalByteDataset(records, sequence_length=64)
        trainer = CountingTrainer()

        trainer.train(
            build_model(),
            create_training_loader(dataset, batch_size=1, shuffle=False),
            TrainingSettings(sequence_length=64, batch_size=1, epochs=2, device="cpu"),
            create_training_loader(dataset, batch_size=1, shuffle=False),
        )

        self.assertEqual(2, trainer.evaluation_calls)


if __name__ == "__main__":
    unittest.main()
