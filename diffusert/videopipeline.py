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
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.models.autoencoder_kl import AutoencoderKL



import onnx
from utilities import TRT_LOGGER, Engine, save_image

from utilities import preprocess_image



class VideoSDPipeline:
    """
    Streamlined tensorrt pipeline for stable diffusion
    """

    def __init__(
        self,
        denoising_steps=50,
        scheduler=EulerAncestralDiscreteScheduler,
        guidance_scale=7.5,
        eta=0.0,
        device="cuda",
        hf_token: str = None,
        verbose=False,
        model_path="runwayml/stable-diffusion-v1-5",
    ):
        """
        Initializes the Diffusion pipeline.
        Args:
            denoising_steps (int):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense of slower inference.
            denoising_fp16 (bool):
                Run the denoising loop (UNet) in fp16 precision.
                When enabled image quality will be lower but generally results in higher throughput.
            guidance_scale (float):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            device (str):
                PyTorch device to run inference. Default: 'cuda'
            output_dir (str):
                Output directory for log files and image artifacts
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
            max_batch_size (int):
                Max batch size for dynamic batch engines.
        """
        # Only supports single image per prompt.
        self.num_images = 1

        self.denoising_steps = denoising_steps
        self.denoising_fp16 = True
        assert guidance_scale > 1.0
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.model_path = model_path
        self.hf_token = hf_token
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = True

        # A scheduler to be used in combination with unet to denoise the encoded image latens.
        # This demo uses an adaptation of LMSDiscreteScheduler or DPMScheduler:
        sched_opts = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False,
        }

        self.scheduler = scheduler.from_config(sched_opts)
        self.previous_prompt = ""
        self.tokenizer = None
        self.models = {
            "clip": CLIP(
                hf_token=hf_token,
                device=device,
                verbose=verbose,
                max_batch_size=1,
            ),
            "unet_fp16": UNet(
                model_path=model_path,
                hf_token=hf_token,
                fp16=True,
                device=device,
                verbose=verbose,
                max_batch_size=1,
            ),
            "vae": VAE(
                hf_token=hf_token,
                device=device,
                verbose=verbose,
                max_batch_size=1,
            )
            # ),
            # "vaeencode": VAEEncode(
            #     hf_token=hf_token,
            #     device=device,
            #     verbose=verbose,
            #     max_batch_size=1,
            # ),
        }

        self.engine = {}
        self.stream = cuda.Stream()


    def teardown(self):
        for engine in self.engine.values():
            del engine
        self.stream.free()
        del self.stream

    def getModelPath(self, name, onnx_dir, opt=True):
        return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")

    def loadEngines(
        self,
        engine_dir="/engines/runwayml/stable-diffusion-v1-5",
    ):
        
        # Build engines
        for model_name, obj in self.models.items():
            engine = Engine(model_name, engine_dir)
            self.engine[model_name] = engine

        # Separate iteration to activate engines
        for model_name, obj in self.models.items():
            self.engine[model_name].activate()
        
        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(1, 384, 512),
                device=self.device,
            )


        gc.collect()

    def loadModules(
        self,
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5",subfolder="vae").to("cuda")

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def infer(
        self,
        prompt,
        negative_prompt=[""],
        image_height=384,
        image_width=512,
        guidance_scale=7.5,
        eta=0.0,
        warmup=False,
        verbose=False,
        seed=None,
        output_dir="static/output",
        num_of_infer_steps=50,
        scheduler=EulerAncestralDiscreteScheduler,
        init_image=None,
        strength=0.7,
        display_timing=False
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
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Enable verbose logging.
        """
        # Process inputs
        batch_size = len(prompt)
        assert len(prompt) == len(negative_prompt)

        ## Number of infer steps
        self.denoising_steps = num_of_infer_steps

        sched_opts = {
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False,
        }

        

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        # Create profiling events
        events = {}
        for stage in ["encode", "clip", "denoise", "vae"]:
            for marker in ["start", "stop"]:
                events[stage + "-" + marker] = cudart.cudaEventCreate()[1]

        # Allocate buffers for TensorRT engine bindings


        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        
        if init_image is not None:
            init_image = init_image.resize((512, 384), resample=Image.Resampling.LANCZOS).convert('RGB')

        # Run Stable Diffusion pipeline
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            # latents need to be generated on the target device
            unet_channels = 4  # unet.in_channels
            latents_shape = (
                batch_size * self.num_images,
                unet_channels,
                latent_height,
                latent_width,
            )
            latents_dtype = torch.float32  # text_embeddings.dtype
            latents = torch.randn(
                latents_shape,
                device=self.device,
                dtype=latents_dtype,
                generator=generator,
            )

            
            self.scheduler.set_timesteps(self.denoising_steps, device=self.device)
            # latents = latents * self.scheduler.init_noise_sigma
            #print(self.scheduler.init_noise_sigma)

            if init_image is not None:
                tail = - int(round(strength * num_of_infer_steps))
                self.scheduler.timesteps = self.scheduler.timesteps[tail:]
                self.scheduler.sigmas = self.scheduler.sigmas[(tail-1):]
                if self.nvtx_profile:
                    nvtx_encode = nvtx.start_range(message="encode", color="green")
                cudart.cudaEventRecord(events["encode-start"], 0)
                init_image = preprocess_image(init_image)
                init_image = init_image.to(self.device)
                #print(init_image)
                #print(init_image.shape)
                # init_image_inp = cuda.DeviceView(
                #     ptr=init_image.data_ptr(),
                #     shape=init_image.shape,
                #     dtype=np.float32,
                # )
                # init_latent = torch.randn(
                #     latents_shape,
                #     device=self.device,
                #     dtype=latents_dtype,
                #     generator=generator,
                # )
                #init_latent = self.runEngine("vaeencode", {"images": init_image_inp})["latent"]
                #print(init_latent_trt.dtype)
                init_latent = self.vae.encode(init_image).latent_dist.sample()

                #print(init_latent_trt)
                #print(init_latent.dtype)
                #print(init_latent)
                
                #init_latent = init_latent_trt
                init_latent = 0.18215 * init_latent
                del init_image
                init_image = True


                latents = self.scheduler.add_noise(init_latent, latents, self.scheduler.timesteps[:1])
                #latents = latents * self.scheduler.sigmas[0]

                cudart.cudaEventRecord(events["encode-stop"], 0)
                if self.nvtx_profile:
                    nvtx.end_range(nvtx_encode)
            #else:
                # Scale the initial noise by the standard deviation required by the scheduler
            
            timesteps = self.scheduler.timesteps

            
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            do_classifier_free_guidance = guidance_scale > 1.0

            if self.nvtx_profile:
                nvtx_clip = nvtx.start_range(message="clip", color="green")

            if self.previous_prompt != prompt:
                self.previous_prompt = prompt

                cudart.cudaEventRecord(events["clip-start"], 0)

                # Tokenize input
                self.text_input_ids = (
                    self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt",
                    )
                    .input_ids.type(torch.int32)
                    .to(self.device)
                )

                # CLIP text encoder
                text_input_ids_inp = cuda.DeviceView(
                    ptr=self.text_input_ids.data_ptr(),
                    shape=self.text_input_ids.shape,
                    dtype=np.int32,
                )
                self.text_embeddings = self.runEngine("clip", {"input_ids": text_input_ids_inp})[
                    "text_embeddings"
                ]

                # Duplicate text embeddings for each generation per prompt
                bs_embed, seq_len, _ = self.text_embeddings.shape
                self.text_embeddings = self.text_embeddings.repeat(1, self.num_images, 1)
                self.text_embeddings = self.text_embeddings.view(
                    bs_embed * self.num_images, seq_len, -1
                )

                

                if do_classifier_free_guidance:
                    uncond_tokens: List[str]
                    if negative_prompt is None:
                        uncond_tokens = [""] * batch_size
                    elif type(prompt) is not type(negative_prompt):
                        raise TypeError(
                            f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                            f" {type(prompt)}."
                        )
                    elif isinstance(negative_prompt, str):
                        uncond_tokens = [negative_prompt]
                    elif batch_size != len(negative_prompt):
                        raise ValueError(
                            f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                            f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                            " the batch size of `prompt`."
                        )
                    else:
                        uncond_tokens = negative_prompt

                    max_length = self.text_input_ids.shape[-1]
                    uncond_input_ids = (
                        self.tokenizer(
                            uncond_tokens,
                            padding="max_length",
                            max_length=max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        .input_ids.type(torch.int32)
                        .to(self.device)
                    )
                    uncond_input_ids_inp = cuda.DeviceView(
                        ptr=uncond_input_ids.data_ptr(),
                        shape=uncond_input_ids.shape,
                        dtype=np.int32,
                    )
                    uncond_embeddings = self.runEngine(
                        "clip", {"input_ids": uncond_input_ids_inp}
                    )["text_embeddings"]

                    # Duplicate unconditional embeddings for each generation per prompt
                    seq_len = uncond_embeddings.shape[1]
                    uncond_embeddings = uncond_embeddings.repeat(1, self.num_images, 1)
                    uncond_embeddings = uncond_embeddings.view(
                        batch_size * self.num_images, seq_len, -1
                    )

                    # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
                    self.text_embeddings = torch.cat([uncond_embeddings, self.text_embeddings])

                if self.denoising_fp16:
                    self.text_embeddings = self.text_embeddings.to(dtype=torch.float16)

            cudart.cudaEventRecord(events["clip-stop"], 0)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_clip)




            extra_step_kwargs = self.prepare_extra_step_kwargs(None, eta)

            cudart.cudaEventRecord(events["denoise-start"], 0)

            for step_index, timestep in enumerate(tqdm.tqdm(timesteps)):
                if self.nvtx_profile:
                    nvtx_latent_scale = nvtx.start_range(
                        message="latent_scale"
                    )
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                # LMSDiscreteScheduler.scale_model_input()
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep
                )
                if self.nvtx_profile:
                    nvtx.end_range(nvtx_latent_scale)

                # predict the noise residual
                if self.nvtx_profile:
                    nvtx_unet = nvtx.start_range(message="unet", color="blue")
                dtype = np.float16 if self.denoising_fp16 else np.float32
                if timestep.dtype != torch.float32:
                    timestep_float = timestep.float()
                else:
                    timestep_float = timestep
                sample_inp = cuda.DeviceView(
                    ptr=latent_model_input.data_ptr(),
                    shape=latent_model_input.shape,
                    dtype=np.float32,
                )
                timestep_inp = cuda.DeviceView(
                    ptr=timestep_float.data_ptr(),
                    shape=timestep_float.shape,
                    dtype=np.float32,
                )
                embeddings_inp = cuda.DeviceView(
                    ptr=self.text_embeddings.data_ptr(),
                    shape=self.text_embeddings.shape,
                    dtype=dtype,
                )
                noise_pred = self.runEngine(
                    "unet_fp16",
                    {
                        "sample": sample_inp,
                        "timestep": timestep_inp,
                        "encoder_hidden_states": embeddings_inp,
                    },
                )["latent"]
                if self.nvtx_profile:
                    nvtx.end_range(nvtx_unet)

                if self.nvtx_profile:
                    nvtx_latent_step = nvtx.start_range(
                        message="latent_step"
                    )
                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, timestep, latents, **extra_step_kwargs
                ).prev_sample


                if self.nvtx_profile:
                    nvtx.end_range(nvtx_latent_step)

            latents = 1.0 / 0.18215 * latents
            cudart.cudaEventRecord(events["denoise-stop"], 0)

            if self.nvtx_profile:
                nvtx_vae = nvtx.start_range(message="vae", color="red")
            cudart.cudaEventRecord(events["vae-start"], 0)
            sample_inp = cuda.DeviceView(
                ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32
            )
            images = self.runEngine("vae", {"latent": sample_inp})["images"]
            cudart.cudaEventRecord(events["vae-stop"], 0)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_vae)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()
            if display_timing:
                print("|------------|--------------|")
                print("| {:^10} | {:^12} |".format("Module", "Latency"))
                print("|------------|--------------|")
                if init_image is not None:
                    print(
                        "| {:^10} | {:>9.2f} ms |".format(
                            "Encode",
                            cudart.cudaEventElapsedTime(
                                events["encode-start"], events["encode-stop"]
                            )[1],
                        )
                    )
                print(
                    "| {:^10} | {:>9.2f} ms |".format(
                        "CLIP",
                        cudart.cudaEventElapsedTime(
                            events["clip-start"], events["clip-stop"]
                        )[1],
                    )
                )
                print(
                    "| {:^10} | {:>9.2f} ms |".format(
                        "UNet x " + str(self.denoising_steps),
                        cudart.cudaEventElapsedTime(
                            events["denoise-start"], events["denoise-stop"]
                        )[1],
                    )
                )
                print(
                    "| {:^10} | {:>9.2f} ms |".format(
                        "VAE",
                        cudart.cudaEventElapsedTime(
                            events["vae-start"], events["vae-stop"]
                        )[1],
                    )
                )
                print("|------------|--------------|")
                print(
                    "| {:^10} | {:>9.2f} ms |".format(
                        "Pipeline", (e2e_toc - e2e_tic) * 1000.0
                    )
                )
                print("|------------|--------------|")

            
            # Save image
            image_name_prefix = (
                "sd-"
                + ("fp16" if self.denoising_fp16 else "fp32")
                + "".join(
                    set(
                        [
                            "-" + prompt[i].replace(" ", "_")[:10]
                            for i in range(batch_size)
                        ]
                    )
                )
                + "-"
            )
            imgs = save_image(images, output_dir, image_name_prefix)

            return imgs
