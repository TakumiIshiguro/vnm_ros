import importlib.util
import os

from PIL import Image


SCRIPT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "evaluate_dataset_predictions.py",
    )
)
SPEC = importlib.util.spec_from_file_location("evaluate_dataset_predictions", SCRIPT_PATH)
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


def test_image_panel_stacks_three_contexts_and_enlarges_observation():
    images = [
        Image.new("RGB", (640, 480), color)
        for color in (
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
        )
    ]

    panel = EVALUATE.prepare_image_panel(images, [10, 11, 12, 13], 160)

    assert len(panel["contexts"]) == 3
    assert [image.height for image, _ in panel["contexts"]] == [160, 160, 160]
    assert panel["observation"].height == 320
    assert panel["height"] == 504
    assert panel["observation_x"] > panel["context_width"]


def test_image_panel_uses_three_most_recent_contexts():
    images = [Image.new("RGB", (40, 30), (index, 0, 0)) for index in range(6)]

    panel = EVALUATE.prepare_image_panel(images, [20, 21, 22, 23, 24, 25], 30)

    assert [index for _, index in panel["contexts"]] == [22, 23, 24]
    assert panel["observation_index"] == 25
