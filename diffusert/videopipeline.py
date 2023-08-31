#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import nvtx
import time
import torch
import cv2
import tensorrt as trt
from utilities import TRT_LOGGER
from stable_diffusion_pipeline import StableDiffusionPipeline
# from sdrefpipeline import StableDiffusionControlNetReferencePipeline
# from diffusers.models import ControlNetModel
# from diffusers import UniPCMultistepScheduler

from PIL import Image






class VideoSDPipeline(StableDiffusionPipeline):
    """
    Application showcasing the acceleration of Stable Diffusion Txt2Img v1.4, v1.5, v2.0, v2.0-base, v2.1, v2.1-base pipeline using NVidia TensorRT w/ Plugins.
    """
    def __init__(
        self,
        scheduler="DDIM",
        *args, **kwargs
    ):
        # self.device = kwargs.get("device", "cuda")
        # print(self.device)
        # self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        # self.pipe = StableDiffusionControlNetReferencePipeline.from_pretrained(
        #         "runwayml/stable-diffusion-v1-5",
        #         controlnet=self.controlnet,
        #         safety_checker=None,
        #         torch_dtype=torch.float16
        #         ).to(self.device)

        # self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        """
        Initializes the Txt2Img Diffusion pipeline.

        Args:
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of the [DPM, LMSD, DDIM, EulerA, PNDM].
        """
        super(VideoSDPipeline, self).__init__(*args, **kwargs, \
            scheduler=scheduler, stages=['clip','unet','vae','vae_encoder'])
    
    def get_canny_filter(self, image, low_threshold=100, high_threshold=200):
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return image

    def get_color_filter(self, cond_image, mask_size=64):
        H, W = cond_image.shape[:2]
        cond_image = cv2.resize(cond_image, (W // mask_size, H // mask_size), interpolation=cv2.INTER_CUBIC)
        color = cv2.resize(cond_image, (W, H), interpolation=cv2.INTER_NEAREST)
        return color

    def get_colorcanny(self, image, mask_size):
        canny_img = self.get_canny_filter(image)

        color_img = self.get_color_filter(image, int(mask_size))

        color_img[np.where(canny_img > 128)] = 255
        return color_img

    def infer(
        self,
        img,
        prompt,
        negative_prompt=["text, distorted face, blurry, bad quality, bad anatomy, low quality"],
        image_height=360,
        image_width=640,
        strength=0.4,
        num_of_infer_steps=20,  
        guidance_scale=7.5,
        seed=42,
        warmup=False,
        verbose=False
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            seed (int):
                Seed for the random generator
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Verbose in logging
        """
        assert len(prompt) == len(negative_prompt)
        img = img.resize((640, 360), resample=Image.Resampling.LANCZOS)

        assert guidance_scale > 1.0
        self.guidance_scale = guidance_scale

        canny_image = np.array(img)

        #low_threshold = 100
        #high_threshold = 200

        canny_image = self.get_canny_filter(canny_image)

        images = self.infertrt(img, num_of_infer_steps, strength, prompt, negative_prompt, canny_image)
        del canny_image
        return images


    def infertrt(self, img, num_of_infer_steps, strength, prompt, negative_prompt, canny_image):

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):

            self.generator.set_state(self.generator_init_state)

            # initialize timesteps
            timesteps, t_start = self.initialize_timesteps(num_of_infer_steps,strength)
            # Pre-initialize latents
            encimg = self.preprocess_image(img, init=True)
            init_latent = self.encode_image(encimg)
            #initialize previous_latents the first time
            if not hasattr(self, 'previous_latents'):
                
                self.previous_latents = init_latent

            latents = init_latent * 0.05 + self.previous_latents * 0.95

            
            e2e_tic = time.perf_counter()

            # CLIP text encoder
            # if prompt is different from the previous prompt, re-encode the prompt

            if not hasattr(self, 'previous_prompt'):
                self.previous_prompt = "asfasdf"
            if prompt != self.previous_prompt:
                text_embeddings = self.encode_prompt(prompt, negative_prompt)


            #torch.cuda.synchronize(self.device)
            # UNet denoiser
            latents = self.denoise_latent(canny_image, init_latent, text_embeddings, timesteps=timesteps, step_offset=1)
            del init_latent
            self.previous_latents = latents


            # VAE decode latent
            images = self.decode_latent(latents)
            del latents
            del timesteps

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()
            #torch.cuda.synchronize(self.device)

            
            images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
            out = []
            for i in range(images.shape[0]):
                out.append(Image.fromarray(images[i]))
            return out
        
            # if not warmup:
            #     self.print_summary(self.denoising_steps, e2e_tic, e2e_toc)
            #     self.save_image(images, 'txt2img', prompt)
