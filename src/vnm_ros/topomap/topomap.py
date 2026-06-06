import os
from typing import List

from PIL import Image as PILImage


def _numeric_image_key(filename: str):
    stem = os.path.splitext(filename)[0]
    try:
        return int(stem)
    except ValueError:
        return stem


class Topomap:
    def __init__(self, directory: str):
        self.directory = directory
        self.images = self._load_images(directory)
        if not self.images:
            raise ValueError(f"No topomap images found in {directory}")

    @staticmethod
    def _load_images(directory: str) -> List[PILImage.Image]:
        filenames = [
            f
            for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        filenames = sorted(filenames, key=_numeric_image_key)
        return [PILImage.open(os.path.join(directory, f)).convert("RGB") for f in filenames]

    def __len__(self) -> int:
        return len(self.images)

    def window(self, start: int, end: int):
        return self.images[start : end + 1]

