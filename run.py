import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image_path = "example.jpg"
image = PIL.Image.open(image_path)
image = PIL.ImageOps.exif_transpose(image)
image = image.convert("RGB")

prompt = "make him look like he was a alive man"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
images[0]
