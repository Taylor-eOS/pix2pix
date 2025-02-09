import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image_path = "example.jpg"
image = PIL.Image.open(image_path)
image = PIL.ImageOps.exif_transpose(image)
image = image.convert("RGB")
prompt = input("Prompt: ")
images = pipe(prompt, image=image, num_inference_steps=5, image_guidance_scale=1).images

output_path = "output_image.jpg"
images[0].save(output_path)
print(f"Image saved to {output_path}")

