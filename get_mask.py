from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
from numpy import asarray
import numpy as np


sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
# TODO: more options to be adjusted: pred_iou_thresh,

image = Image.open("./datasets/loco/breakfast_box/train/good/000.png")
image_np = asarray(image)

masks = mask_generator.generate(image_np)

# overlay masks to the image
if len(masks) > 0:
    sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    image.convert("RGBA")
    width = sorted_anns[0]["segmentation"].shape[0]
    height = sorted_anns[0]["segmentation"].shape[1]

    for ann in sorted_anns:
        overlay = np.ones(
            (
                width,
                height,
                4,
            )
        )
        overlay[:, :, 3] = 0
        overlay[ann["segmentation"]] = np.concatenate(
            [np.random.random(3), [0.35]]
        )  # 0.35 is opacity

        image.paste(overlay, (0, 0), overlay)

image.save(f"result.png", "PNG")
