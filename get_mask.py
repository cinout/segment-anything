from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
from numpy import asarray


sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
# TODO: more options to be adjusted: pred_iou_thresh,

image = Image.open("./datasets/loco/breakfast_box/train/good/000.png")
image_np = asarray(image)

print(f">>> image_np: {image_np.shape}")
masks = mask_generator.generate(image_np)
print(masks)
