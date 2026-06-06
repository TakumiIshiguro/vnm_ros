from typing import List

import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image

IMAGE_ASPECT_RATIO = 4 / 3


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding in ("bgr8", "8UC3"):
        img = img[:, :, ::-1]
    if img.shape[-1] == 1:
        img = img[:, :, 0]
    return PILImage.fromarray(img)


def pil_to_msg(pil_img: PILImage.Image, encoding: str = "rgb8") -> Image:
    img = np.asarray(pil_img.convert("RGB"))
    msg = Image()
    msg.encoding = encoding
    msg.height, msg.width = img.shape[:2]
    msg.step = msg.width * 3
    msg.data = img.tobytes()
    return msg


def transform_images(pil_imgs, image_size: List[int], center_crop: bool = False):
    import torch
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensors = []
    for pil_img in pil_imgs:
        pil_img = pil_img.convert("RGB")
        if center_crop:
            w, h = pil_img.size
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        tensors.append(torch.unsqueeze(transform(pil_img.resize(image_size)), 0))
    return torch.cat(tensors, dim=1)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
