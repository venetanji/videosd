import torch
import xformers
import triton
from sfast.compilers.stable_diffusion_pipeline_compiler import compile, CompilationConfig
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
import os, contextlib
import sys

# download an image
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
np_image = np.array(image)
# get canny image
np_image = cv2.Canny(np_image, 100, 200)
np_image = np_image[:, :, None]
np_image = np.concatenate([np_image, np_image, np_image], axis=2)
canny_image = Image.fromarray(np_image)


def load_model():
    controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained("SG161222/Realistic_Vision_V5.1_noVAE", 
                                                                controlnet=controlnet, torch_dtype=torch.float16)

    model.safety_checker = None
    model.to(torch.device('cuda'))
    return model

model = load_model()

#generator = torch.manual_seed(0)

# image = pipe(
#     "futuristic-looking woman",
#     num_inference_steps=50,
#     generator=generator,
#     image=image,
#     control_image=canny_image,
# ).images[0]

# generator = torch.manual_seed(0)


# image = pipe(
#     "futuristic-looking woman",
#     num_inference_steps=50,
#     generator=generator,
#     image=image,
#     control_image=canny_image,
# ).images[0]

# generate image


config = CompilationConfig.Default()
config.enable_xformers = True
config.enable_triton = True
config.enable_cuda_graph = True

# xformers and triton are suggested for achieving best performance.
# It might be slow for triton to generate, compile and fine-tune kernels.

# CUDA Graph is suggested for small batch sizes.
# After capturing, the model only accepts one fixed image size.
# If you want the model to be dynamic, don't enable it.
compiled_model = compile(model, config)

kwarg_inputs = dict(
    prompt=
    '(masterpiece:1,2), best quality, masterpiece, best detail face, lineart, monochrome, a beautiful girl',
    height=512,
    width=512,
    num_inference_steps=50,
    num_images_per_prompt=1,
    image=canny_image
)

# NOTE: Warm it up.
# The first call will trigger compilation and might be very slow.
# After the first call, it should be very fast.
# output_image = compiled_model(**kwarg_inputs).images[0]

compiled_model(**kwarg_inputs).images[0]


compiled_model(**kwarg_inputs).images[0]
