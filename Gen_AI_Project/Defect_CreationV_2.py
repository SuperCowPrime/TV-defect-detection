import torch
import os
import random
from PIL import Image
import concurrent.futures
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

batchSize = 4

baseFolder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
maskFolder = os.path.join(baseFolder, "masks")
output_folder = os.path.join(baseFolder, "defected_tvs")
os.makedirs(output_folder, exist_ok=True)

defectPrompts = {
    "spiderweb": (
        "hairline spiderweb crack on the glass screen,fine branching fractures",
        0.7
    ),

    "scratch": (
        "multiple surface scratches on the tv glass, "
        "realistic worn screen",
        0.99
    ),

    "shattered_corner": (
        "cracked corner of the tv screen, diagonal fracture lines running inward from the edge, tight cluster of cracks near the corner,"
        "realistic lcd glass damage",
        0.9
    ),

    "puncture": (
        "bullet hole in tv screen, crushed glass center, thick radial cracks, "
        "white shattered edges, high visibility damage",
        0.7
    )
}

negativePrompt = "blur, low quality, reflection of photographer, text, watermark, healthy screen, cartoon, painting"

print("Loading model")
pipe = AutoPipelineForInpainting.from_pretrained(
    "OzzyGT/RealVisXL_V4.0_inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

pipe.enable_vae_tiling() 

def chunkList(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

def saveImagesAsync(images, batchData, defectName, outRoot):
    specificFolder = os.path.join(outRoot, defectName)
    os.makedirs(specificFolder, exist_ok=True)
    for resultImg, item in zip(images, batchData):
        savePath = os.path.join(specificFolder, item['name'])
        resultImg.save(savePath)
        print(f"Saved {item['name']} to '{defectName}'")

validPairs = []
rawFiles = [f for f in os.listdir(baseFolder) if f.endswith(".png") and "mask" not in f]
for filename in rawFiles:
    maskPath = os.path.join(maskFolder, filename.replace(".png", "_mask.png"))
    if os.path.exists(maskPath):
        validPairs.append({'img': os.path.join(baseFolder, filename), 'mask': maskPath, 'name': filename})

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

print("Starting generation")
for batch in chunkList(validPairs[:50], batchSize):

    batchImages = [load_image(item['img']) for item in batch]
    batchMasks = [load_image(item['mask']) for item in batch]

    defectName, (promptText, specificStrength) = random.choice(list(defectPrompts.items()))

    results = pipe(
        prompt=[promptText] * len(batchImages),
        negative_prompt=[negativePrompt] * len(batchImages),
        image=batchImages,
        mask_image=batchMasks,
        height=720, width=1280,
        num_inference_steps=20,
        guidance_scale=8,
        strength=specificStrength
    ).images

    executor.submit(saveImagesAsync, results, batch, defectName, output_folder)

executor.shutdown(wait=True)
print("Done")