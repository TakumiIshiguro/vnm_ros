import os
import shutil
import time


class TopomapBuilder:
    def __init__(self, output_dir: str, dt: float, overwrite: bool = True):
        self.output_dir = output_dir
        self.dt = dt
        self.overwrite = overwrite
        self.index = 0
        self.last_saved_time = float("-inf")

    def prepare(self):
        if os.path.isdir(self.output_dir) and self.overwrite:
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def maybe_save(self, image, stamp: float = None) -> bool:
        now = time.time() if stamp is None else stamp
        if now - self.last_saved_time < self.dt:
            return False
        image.save(os.path.join(self.output_dir, f"{self.index}.png"))
        self.index += 1
        self.last_saved_time = now
        return True
