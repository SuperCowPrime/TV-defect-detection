import torch
import os
import random
from PIL import Image
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

# --- CONFIG ---
base_folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
mask_folder = os.path.join(base_folder, "masks")
output_folder = os.path.join(base_folder, "defected_tvs")
os.makedirs(output_folder, exist_ok=True)

# Prompts for different defects
defect_prompts = [
    "huge spiderweb crack in the center of the screen, impact point, shattered glass texture",
    "deep long scratches across the display, gouged surface, abrasion marks, rough texture",
    "shattered corner of the screen, missing glass pieces, jagged sharp edges",
    "small puncture hole in glass screen, precise impact, radial cracks around hole",
    "heavy blunt force damage, crushed glass crater, destroyed panel surface, caved in screen",
    "single long vertical crack running down the entire screen, deep fissure, split glass"
]

negative_prompt = "blur, low quality, reflection of photographer, text, watermark, healthy screen"

# --- LOAD MODEL ---
print("Loading Inpainting Model...")
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# --- PROCESSING LOOP ---
# Get list of images that have a corresponding mask
image_files = [f for f in os.listdir(base_folder) if f.endswith(".png") and "mask" not in f]

# Limit for testing (remove [:10] to run all)
for i, filename in enumerate(image_files[:10]):

    # 1. Load Original and Mask
    img_path = os.path.join(base_folder, filename)
    mask_path = os.path.join(mask_folder, filename.replace(".png", "_mask.png"))

    if not os.path.exists(mask_path):
        print(f"Skipping {filename}, mask not found.")
        continue

    original_image = load_image(img_path)
    mask_image = load_image(mask_path)

    # 2. Select Random Defect
    prompt = random.choice(defect_prompts)

    # 3. Inpaint
    # strength=0.99 means "Destroy the black pixels and build glass from scratch"
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=original_image,
        mask_image=mask_image,
        height=720,
        width=1280,
        num_inference_steps=30,
        strength=0.5,
        guidance_scale=7.5
    ).images[0]

    # 4. Save
    save_name = f"defect_{filename}"
    result.save(os.path.join(output_folder, save_name))

    print(f"[{i + 1}] Created defect: {prompt[:20]}... -> {save_name}")

print("Done! Check the 'defected_tvs' folder.")