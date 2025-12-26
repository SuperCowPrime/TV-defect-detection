import torch
import random
import os
import time
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerDiscreteScheduler


save_folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
os.makedirs(save_folder, exist_ok=True)
print(f"Images will be saved to: {save_folder}")
BATCH_SIZE = 7

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)


pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0_Lightning",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

total_Images_Wanted = 200*BATCH_SIZE #change the number to the desired amount of pictures wanted.

angles = ["front view", "close up", "side view", "slightly high angle","slightly low angle","top down view"]


lighting_Conditions =[
    "bright fluorescent factory lights",
    "dim moody industrial lighting",
    "harsh directional spotlight with glare",
    "soft diffused studio lighting",
    "natural window light reflection"
]

negative_prompt = (
            "white screen, glowing screen, blue screen, picture on screen, content, image, "
            "emission, light source, reflection, "
            "clutter, wires, cables, trash, workers, humans, people, "
            "text, logo, brand name, watermark,screen writing, "
            "complex machinery, messy"
        )

num_Batches = total_Images_Wanted//BATCH_SIZE

print(f"Starting FAST LIGHTNING generation...")

for batch_index in range(num_Batches):
    prompts = []
    seeds = []

    for i in range(BATCH_SIZE):
        angle = random.choice(angles)
        lighting = random.choice(lighting_Conditions)
        seed = random.randint(0, 2 ** 32 - 1)


        prompt = (
            f"Industrial product render of a single black flat screen TV. "
            f"{angle},{lighting}."
            f"The screen is turned off, matte black, obsidian, void, no reflection. "
            f"Background is a blurred modern factory interior. "
            f"Minimalist, clean, 8k uhd."
        )
        prompts.append(prompt)
        seeds.append(seed)

    generators = [torch.Generator("cuda").manual_seed(s) for s in seeds]
    negative_prompts = [negative_prompt] * BATCH_SIZE


    start_time = time.time()

    image = pipe(
        prompt=prompts,
        negative_prompt=negative_prompts,
        height=720,
        width=1280,
        num_inference_steps=7,
        guidance_scale=2,
        generator=generators
    ).images

    end_time = time.time()
    batch_time = end_time - start_time

    for index, image in enumerate(image):
        global_count = (batch_index * BATCH_SIZE) + index
        seed = seeds[index]

        filename = f"tv_fast_{global_count:04d}_{seed}.png"
        image.save(os.path.join(save_folder, filename))

    print(f"[{batch_index + 1}/{num_Batches}] Batch finished in {batch_time:.2f}s")
print("Done!")