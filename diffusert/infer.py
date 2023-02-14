import argparse
import gc
from pathlib import Path
import inspect
import io
import os
import shutil
import time
from typing import List

import numpy as np
import nvtx
import tensorrt as trt
import torch
import tqdm
from cuda import cudart
from polygraphy import cuda
from models import CLIP, VAE, UNet, VAEEncode
import onnx
from utilities import TRT_LOGGER, Engine
from PIL import Image
from transformers import CLIPTokenizer

from diffusers.models.autoencoder_kl import AutoencoderKL

from videopipeline import VideoSDPipeline
import onnx

from utilities import preprocess_image

os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

def parseArgs():
    "Parse command line arguments"

    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Demo")
    # Stable Diffusion configuration
    parser.add_argument(
        "--prompt", nargs="*", default="A drawing of a castle in the clouds", help="Text prompt(s) to guide image generation"
    )
    parser.add_argument(
        "--init-image", nargs="*", help="Init image"
    )
    parser.add_argument(
        "--strength", nargs="*", help="Denoising strength when using init image"
    )
    parser.add_argument(
        "--negative-prompt",
        nargs="*",
        default=[""],
        help="The negative prompt(s) to guide the image generation",
    )
    parser.add_argument(
        "--repeat-prompt",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Number of times to repeat the prompt (batch size multiplier)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of image to generate (must be multiple of 8)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Height of image to generate (must be multiple of 8)",
    )
    # parser.add_argument('--num-images', type=int, default=1, help="Number of images to generate per prompt")
    parser.add_argument(
        "--denoising-steps", type=int, default=50, help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance-scale", type=int, default=7, help="Guidance scale"
    )

    parser.add_argument(
        "--denoising-prec",
        type=str,
        default="fp16",
        choices=["fp32", "fp16"],
        help="Denoiser model precision",
    )

    # TensorRT engine build
    parser.add_argument(
        "--model-path",
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace Model path",
    )
    parser.add_argument(
        "--engine-dir", default="/engines", help="Output directory for TensorRT engines"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random generator to get consistent results",
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for logs and image artifacts",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )

    return parser.parse_args()



def load_trt(model='runwayml/stable-diffusion-v1-5'):
    global trt_model
    global loaded_model
    # if a model is already loaded, remove it from memory
    try:
        trt_model.teardown()
    except:
        pass

    args = parseArgs()
    engines_dir = Path(args.engine_dir)
    model_engine_dir = engines_dir/ model

    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    # Initialize demo
    trt_model = VideoSDPipeline(
        model_path=model,
        denoising_steps=args.denoising_steps,
        verbose=args.verbose
    )

    trt_model.loadEngines(model_engine_dir)
    trt_model.loadModules()
    loaded_model = model


def infer_trt(
    saving_path="output",
    model='runwayml/stable-diffusion-v1-5',
    prompt='A photo of a cat',
    neg_prompt="",
    img_height=384,
    img_width=512,
    num_inference_steps=20,
    guidance_scale=7,
    num_images_per_prompt=1,
    seed=None,
):
    global trt_model
    global loaded_model
    print("[I] Initializing StableDiffusion demo with TensorRT Plugins")
    args = parseArgs()

    args.output_dir = saving_path
    args.prompt = [prompt]
    args.model_path = model
    args.height = img_height
    args.width = img_width
    args.repeat_prompt = num_images_per_prompt
    args.denoising_steps = num_inference_steps
    args.seed = seed
    args.guidance_scale = guidance_scale
    args.negative_prompt = [neg_prompt]

    print("Seed :", args.seed)

    # Process prompt
    if not isinstance(args.prompt, list):
        raise ValueError(
            f"`prompt` must be of type `str` or `str` list, but is {type(args.prompt)}"
        )
    # print('String :', args.prompt, type(args.prompt))
    prompt = args.prompt * args.repeat_prompt

    if not isinstance(args.negative_prompt, list):
        raise ValueError(
            f"`--negative-prompt` must be of type `str` or `str` list, but is {type(args.negative_prompt)}"
        )
    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt

    # Validate image dimensions
    image_height = args.height
    image_width = args.width
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(
            f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}."
        )
    load_trt(model)

    images = trt_model.infer(
        prompt,
        negative_prompt,
        args.height,
        args.width,
        guidance_scale=args.guidance_scale,
        num_of_infer_steps=args.denoising_steps,
        verbose=False,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    gc.collect()
    return images



if __name__ == "__main__":

    print("[I] Initializing StableDiffusion demo with TensorRT Plugins")
    args = parseArgs()
    infer_trt(
        saving_path=args.output_dir,
        model=args.model_path,
        prompt=args.prompt[0],
        neg_prompt=args.negative_prompt[0],
        img_height=args.height,
        img_width=args.width,
        num_inference_steps=args.denoising_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.repeat_prompt,
        seed=args.seed,
    )

