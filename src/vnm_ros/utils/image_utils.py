from typing import List

import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image


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


def center_crop_resize(pil_img: PILImage.Image, image_size: List[int]) -> PILImage.Image:
    pil_img = pil_img.convert("RGB")
    width, height = pil_img.size
    target_ratio = 4.0 / 3.0
    if width / height > target_ratio:
        crop_width = int(height * target_ratio)
        left = (width - crop_width) // 2
        pil_img = pil_img.crop((left, 0, left + crop_width, height))
    else:
        crop_height = int(width / target_ratio)
        top = (height - crop_height) // 2
        pil_img = pil_img.crop((0, top, width, top + crop_height))
    return pil_img.resize(tuple(image_size))


def transform_images(pil_imgs, image_size: List[int], center_crop: bool = True):
    import torch
    from torchvision import transforms

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
            if pil_img.size != tuple(image_size):
                pil_img = center_crop_resize(pil_img, image_size)
        else:
            pil_img = pil_img.resize(tuple(image_size))
        tensors.append(torch.unsqueeze(transform(pil_img), 0))
    return torch.cat(tensors, dim=1)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
