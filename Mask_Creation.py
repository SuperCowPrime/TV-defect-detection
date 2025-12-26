import torch
import os
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from segment_anything import sam_model_registry, SamPredictor



image_folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
mask_output_folder = os.path.join(image_folder, "masks")
os.makedirs(mask_output_folder, exist_ok=True)
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda"



processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", do_image_splitting=False)
detector = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
detector.eval()


sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)


files = [f for f in os.listdir(image_folder) if f.endswith(".png")]

for i, filename in enumerate(files):
    img_path = os.path.join(image_folder, filename)
    pil_image = Image.open(img_path).convert("RGB")


    texts = [["flat black screen,television screen,display"]]
    inputs = processor(text=texts, images=pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = detector(**inputs)


    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

    if len(results["boxes"]) == 0:
        print(f"[{i}] No TV found in {filename}, skipping.")
        continue

    boxes = results["boxes"]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    best_idx = areas.argmax()
    best_box = boxes[best_idx].cpu().numpy()


    image_np = np.array(pil_image)
    predictor.set_image(image_np)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=best_box[None, :],
        multimask_output=False,
    )


    binary_mask = masks[0].astype(np.uint8) * 255


    mask_image = Image.fromarray(binary_mask)
    mask_filename = filename.replace(".png", "_mask.png")
    mask_image.save(os.path.join(mask_output_folder, mask_filename))

    print(f"[{i + 1}/{len(files)}] Mask saved: {mask_filename}")

print("Done. Masks are ready for inpainting.")

