import torch
import os
import random
from PIL import Image
import concurrent.futures
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
BATCH_SIZE = 4

# --- CONFIG ---
base_folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
mask_folder = os.path.join(base_folder, "masks")
output_folder = os.path.join(base_folder, "defected_tvs")
os.makedirs(output_folder, exist_ok=True)

# Prompts for different defects
defect_prompts = {
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


negative_prompt = "blur, low quality, reflection of photographer, text, watermark, healthy screen, cartoon, painting"

# --- LOAD MODEL (NEW MODEL) ---
print("Loading RealVisXL V4.0 Inpainting Model...")
pipe = AutoPipelineForInpainting.from_pretrained(
    "OzzyGT/RealVisXL_V4.0_inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# 2. OPTIMIZATION: Memory efficient attention and Tiling
pipe.enable_vae_tiling()  # Saves VRAM, prevents OOM on larger batches


# --- HELPER FUNCTIONS ---
def chunk_list(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


def save_images_async(images, batch_data, defect_name, out_root):
    """Saves images in a separate thread to unblock GPU."""
    specific_folder = os.path.join(out_root, defect_name)
    os.makedirs(specific_folder, exist_ok=True)
    for result_img, item in zip(images, batch_data):
        save_path = os.path.join(specific_folder, item['name'])
        result_img.save(save_path)
        print(f"Saved {item['name']} to '{defect_name}'")


# --- MAIN LOOP ---
valid_pairs = []
raw_files = [f for f in os.listdir(base_folder) if f.endswith(".png") and "mask" not in f]
for filename in raw_files:
    mask_path = os.path.join(mask_folder, filename.replace(".png", "_mask.png"))
    if os.path.exists(mask_path):
        valid_pairs.append({'img': os.path.join(base_folder, filename), 'mask': mask_path, 'name': filename})

# ThreadPool for non-blocking saving
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

print("Starting generation...")
for batch in chunk_list(valid_pairs[:50], BATCH_SIZE):
    # Load images
    batch_images = [load_image(item['img']) for item in batch]
    batch_masks = [load_image(item['mask']) for item in batch]

    defect_name, (prompt_text, specific_strength) = random.choice(list(defect_prompts.items()))

    # Run Pipeline
    results = pipe(
        prompt=[prompt_text] * len(batch_images),
        negative_prompt=[negative_prompt] * len(batch_images),
        image=batch_images,
        mask_image=batch_masks,
        height=720, width=1280,
        num_inference_steps=20,  # OPTIMIZATION: Reduced from 30 to 20
        guidance_scale=8,
        strength=specific_strength
    ).images

    # 3. OPTIMIZATION: Offload saving to thread
    executor.submit(save_images_async, results, batch, defect_name, output_folder)

# Clean up
executor.shutdown(wait=True)
print("Done!")