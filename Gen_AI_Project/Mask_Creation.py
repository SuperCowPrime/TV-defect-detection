import torch
import os
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from segment_anything import sam_model_registry, SamPredictor

imageFolder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
maskOutputFolder = os.path.join(imageFolder, "masks")
os.makedirs(maskOutputFolder, exist_ok=True)
samCheckpoint = "sam_vit_h_4b8939.pth"
device = "cuda"

print("Loading Detactor")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", do_image_splitting=False)
detector = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
detector.eval()

print("Loading SAM (Segmenter)...")
sam = sam_model_registry["vit_h"](checkpoint=samCheckpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

files = [f for f in os.listdir(imageFolder) if f.endswith(".png")]

for i, filename in enumerate(files):
    imgPath = os.path.join(imageFolder, filename)
    pilImage = Image.open(imgPath).convert("RGB")

    texts = [["flat black screen,television screen,display"]]
    inputs = processor(text=texts, images=pilImage, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = detector(**inputs)

    target_sizes = torch.tensor([pilImage.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

    if len(results["boxes"]) == 0:
        print(f"[{i}] No TV found in {filename}, skipping.")
        continue

    boxes = results["boxes"]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    bestIdx = areas.argmax()
    bestBox = boxes[bestIdx].cpu().numpy()

    imageNp = np.array(pilImage)
    predictor.set_image(imageNp)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bestBox[None, :],
        multimask_output=False,
    )

    binaryMask = masks[0].astype(np.uint8) * 255

    maskImage = Image.fromarray(binaryMask)
    maskFilename = filename.replace(".png", "_mask.png")
    maskImage.save(os.path.join(maskOutputFolder, maskFilename))

    print(f"[{i + 1}/{len(files)}] Mask saved: {maskFilename}")

print("Done")

