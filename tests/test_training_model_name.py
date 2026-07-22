import importlib.util
import os
import unittest


TRAIN_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts", "train.py")
)
SPEC = importlib.util.spec_from_file_location("vnm_train", TRAIN_SCRIPT)
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class TrainingModelNameTest(unittest.TestCase):
    def test_prefixes_best_and_epoch_checkpoints(self):
        config = TRAIN.train_cfg_for_epoch_count(
            {"training": {"model_name": "encoder_lr", "epoch": [5]}},
            5,
        )

        training = config["training"]
        self.assertEqual(training["best_checkpoint_name"], "encoder_lr_best.pth")
        self.assertEqual(
            training["final_checkpoint_name"], "encoder_lr_epoch005.pth"
        )

    def test_empty_name_preserves_existing_checkpoint_names(self):
        config = TRAIN.train_cfg_for_epoch_count(
            {"training": {"model_name": "", "epoch": [10]}},
            10,
        )

        training = config["training"]
        self.assertEqual(training["best_checkpoint_name"], "best.pth")
        self.assertEqual(training["final_checkpoint_name"], "epoch010.pth")

    def test_rejects_path_characters(self):
        with self.assertRaisesRegex(ValueError, "training.model_name"):
            TRAIN.configured_model_name({"model_name": "../outside"})


if __name__ == "__main__":
    unittest.main()
