from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
from numpy import asarray
import numpy as np
import time
import json
import random, os
from datetime import datetime


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"output_{timestamp}"

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(
    sam
)  # TODO: more options to be adjusted: pred_iou_thresh,


categories = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

os.makedirs(output_folder)

for category in categories:
    image = Image.open(f"./datasets/loco/{category}/train/good/000.png")
    image_np = asarray(image)
    masks = mask_generator.generate(image_np)

    # with open("seg_data.npy", "wb") as f:
    #     np.save(f, segmentation_results)

    # with open("seg_data.npy", "rb") as f:
    #     segmentation_results = np.load(f)

    if len(masks) > 0:
        # masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
        segmentation_results = np.array([item["segmentation"] for item in masks])

        image = image.convert("RGBA")
        height = segmentation_results[0].shape[0]
        width = segmentation_results[0].shape[1]

        for i, seg in enumerate(segmentation_results[:20]):
            mask = np.zeros(
                (
                    height,
                    width,
                    4,
                ),
                dtype=np.uint8,  # important, otherwise there will be errors
            )

            mask[seg] = [
                random.randint(1, 255),
                random.randint(1, 255),
                random.randint(1, 255),
                89,
            ]

            mask = Image.fromarray(mask)
            if i <= 10:
                mask.save(f"{output_folder}/{category}_mask_{i}.png", "PNG")

            image.paste(mask, (0, 0), mask)

    image.save(f"{output_folder}/{category}_result.png", "PNG")
