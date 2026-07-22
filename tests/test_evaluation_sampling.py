import importlib.util
import os
import unittest

import numpy as np


EVALUATION_SCRIPT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "evaluate_dataset_predictions.py",
    )
)
SPEC = importlib.util.spec_from_file_location("vnm_evaluation", EVALUATION_SCRIPT)
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


class DatasetStub:
    def __len__(self):
        return 60

    def sample_cmd_dir_labels(self):
        return np.repeat(np.arange(3), 20)


class EvaluationSamplingTest(unittest.TestCase):
    def test_random_selection_is_reproducible_and_balanced(self):
        dataset = DatasetStub()

        first = EVALUATION.selected_sample_indices(dataset, 5, [], sample_seed=7)
        repeated = EVALUATION.selected_sample_indices(dataset, 5, [], sample_seed=7)
        changed = EVALUATION.selected_sample_indices(dataset, 5, [], sample_seed=8)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        labels = dataset.sample_cmd_dir_labels()[first]
        np.testing.assert_array_equal(np.bincount(labels, minlength=3), [5, 5, 5])

    def test_explicit_indices_ignore_sample_seed(self):
        selected = EVALUATION.selected_sample_indices(
            DatasetStub(), 5, [2, 21, 44], sample_seed=99
        )

        self.assertEqual(selected, [2, 21, 44])


if __name__ == "__main__":
    unittest.main()
