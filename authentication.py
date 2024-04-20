from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler
import torch

pipe = StableDiffusionXLPipeline.from_single_file("https://huggingface.co/martyn/sdxl-turbo-mario-merge-top-rated/blob/main/topRatedTurboxlLCM_v10.safetensors")
# pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Load the LoRA
pipe.load_lora_weights('ntc-ai/SDXL-LoRA-slider.2000s-indie-comic-art-style', weight_name='2000s indie comic art style.safetensors', adapter_name="2000s indie comic art style")

# Activate the LoRA
pipe.set_adapters(["2000s indie comic art style"], adapter_weights=[2.0])

prompt = "medieval rich kingpin sitting in a tavern, 2000s indie comic art style"
negative_prompt = "nsfw"
width = 512
height = 512
num_inference_steps = 10
guidance_scale = 2
image = pipe(prompt, negative_prompt=negative_prompt, width=width, height=height, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
image.save('result.png')
