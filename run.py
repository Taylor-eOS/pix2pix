import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import ImageFilter, ImageEnhance

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

image_path = "image.jpg"
image = PIL.Image.open(image_path)
image = PIL.ImageOps.exif_transpose(image)  
image = image.convert("RGB")  

res = 256
new_size = (res, res)  
image = image.resize(new_size)

print(f"Resolution is {res}x{res}, scale your image accordingly")
prompt = input("Prompt: ")
num_inference_steps = int(input("Number of inference steps (default: 25): ") or 25)
image_guidance_scale = float(input("Image guidance scale (default: 1.5): ") or 1.5)

images = pipe(prompt, image=image, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale).images

sharpened_image = images[0].filter(ImageFilter.SHARPEN)
enhancer = ImageEnhance.Contrast(sharpened_image)
enhanced_image = enhancer.enhance(1.2)  

output_path = "output_image.jpg"  
enhanced_image.save(output_path)
print(f"Image saved to {output_path}")
