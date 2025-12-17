import torch
import random
import os
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler

from Generation import total_Image_Wanted

save_folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
os.makedirs(save_folder, exist_ok=True)
print(f"Images will be saved to: {save_folder}")


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)


total_Images_Wanted = 50

angles = ["front view", "isometric view", "side view", "slightly high angle"]

print(f"Starting SDXL generation...")

for i in range(total_Images_Wanted):
    angle = random.choice(angles)
    seed = random.randint(0, 2 ** 32 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)

    
    prompt = (
        f"Product photography of a generic black flat screen TV standing on a simple white table. "
        f"{angle}. The screen is turned off and purely black. "
        f"Solid light grey background. Minimalist. 8k uhd, high quality."
    )

    # 2. NEGATIVE PROMPT
    negative_Prompt = (
        "human, person, woman, man, face, reflection of person, "
        "furniture, shelves, cabinets, lamps, tripods, "
        "clutter, messy, complex background, "
        "text, logo, watermark"
    )

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_Prompt,
        height=720,
        width=1280,  
        num_inference_steps=25,
        guidance_scale=7.0,
        generator=generator
    ).images[0]

    filename = f"tv_sdxl_{i:04d}_{seed}.png"
    full_path = os.path.join(save_folder, filename)
    image.save(full_path)

    print(f"[{i + 1}/{total_Images_Wanted}] Saved: {filename}")

print("Done!")