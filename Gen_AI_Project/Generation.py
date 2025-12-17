import torch
import os
import random
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import DPMSolverMultistepScheduler

save_Folder = r"C:\Users\amitw\OneDrive\Desktop\Tv_Dataset"
os.makedirs(save_Folder, exist_ok=True)
print(f"Images will be saved to: {save_Folder}")

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)

model_Id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

pipe = StableDiffusionPipeline.from_pretrained(
    model_Id,
    vae=vae,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.to("cuda")

total_Image_Wanted = 10

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

angles = [
    "front view",
    "isometric view",
    "high angle shot looking down",
    "low angle shot",
    "side profile view",
    "close up of screen corner"
]
lighting_Conditions =[
    "bright fluorescent factory lights",
    "dim moody industrial lighting",
    "harsh directional spotlight with glare",
    "soft diffused studio lighting",
    "natural window light reflection"
]


negative_Prompt = (
        "(woman:1.5), (man:1.5), (human:1.5), (person:1.5), (face:1.5), "
        "(model:1.4), (girl:1.4), (boy:1.4), "
        "zoomed in, close up, macro, cropped, cut off,white screen, "
        "tray, box, container, basket,screen on, "
        "living room, bedroom, furniture, sofa, couch, "
        "text, logo, watermark, "
        "clutter, complex background, messy"
    )

print(f"Starting mass generation of {total_Image_Wanted} images...")

for i in range(total_Image_Wanted):
    angle = random.choice(angles)
    lighting_Conditions = random.choice(lighting_Conditions)
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)
    ##guidance = random.uniform(7.0, 8.5)
    prompt = (
        f"medium shot of a (single black flat screen TV:1.4),(rectangular screen:1.2), standing upright on a table, "
        f"{angle}, {lighting_Conditions}, "
        f"screen is turned off and pure black, "
        f"clean grey studio background, minimalist, "
        f"product photography, 4k, sharp focus on the tv frame"
    )

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_Prompt,
        height = 512,
        width = 768,
        num_inference_steps=25,
        guidance_scale=15.0,
        generator=generator
    ).images[0]

    filename = f"tv_gen_{i:04d}_{seed}.png"
    full_Path = os.path.join(save_Folder, filename)
    image.save(full_Path)


    if i % 10 == 0:
        print(f"Progress: {i}/{total_Image_Wanted} images saved.")