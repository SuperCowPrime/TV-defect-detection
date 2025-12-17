import torch
import random
import os
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler


save_folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
os.makedirs(save_folder, exist_ok=True)
print(f"Images will be saved to: {save_folder}")


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)


pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0_Lightning",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")


total_Images_Wanted = 2930

angles = ["front view", "close up", "side view", "slightly high angle","slightly low angle","top down view"]


lighting_Conditions =[
    "bright fluorescent factory lights",
    "dim moody industrial lighting",
    "harsh directional spotlight with glare",
    "soft diffused studio lighting",
    "natural window light reflection"
]

print(f"Starting FAST LIGHTNING generation...")

for i in range(total_Images_Wanted):
    angle = random.choice(angles)
    lighting_Conditions = random.choice(lighting_Conditions)
    seed = random.randint(0, 2 ** 32 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)

    prompt = (
        f"Industrial product render of a single black flat screen TV. "
        f"{angle},{lighting_Conditions}."
        f"The screen is turned off, matte black, obsidian, void, no reflection. "
        f"Background is a blurred modern factory interior. "
        f"Minimalist, clean, 8k uhd."
    )

    negative_prompt = (
        "white screen, glowing screen, blue screen, picture on screen, content, image, "
        "emission, light source, reflection, "
        "clutter, wires, cables, trash, workers, humans, people, "
        "text, logo, brand name, watermark,screen writing, "
        "complex machinery, messy"
    )

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=720,
        width=1280,
        num_inference_steps=7,
        guidance_scale=2,
        generator=generator
    ).images[0]

    filename = f"tv_fast_{i + 70:04d}_{seed}.png"
    full_path = os.path.join(save_folder, filename)
    image.save(full_path)

    print(f"[{i + 1}/{total_Images_Wanted}] Saved: {filename}")

print("Done!")
