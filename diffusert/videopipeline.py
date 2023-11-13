import numpy as np
import time
import torch

#from .lcm.lcm_pipeline import LatentConsistencyModelPipeline
#from lcm.lcm_i2i_pipeline import LatentConsistencyModelImg2ImgPipeline
from lcm.lcm_reference_pipeline import LatentConsistencyModelPipeline_reference, LCMScheduler_X
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
from lcm.lcm_controlnet import LatentConsistencyModelPipeline_controlnet
from lcm.canny_gpu import SobelOperator 
from diffusers import AutoencoderTiny, ControlNetModel
import ray
from PIL import Image


@ray.remote(num_gpus=1)
class VideoSDPipeline:
    """
    Application showcasing the acceleration of Stable Diffusion Txt2Img v1.4, v1.5, v2.0, v2.0-base, v2.1, v2.1-base pipeline using NVidia TensorRT w/ Plugins.
    """
    def __init__(
        self,
        *args, **kwargs
    ):
        self.device = kwargs.get("device", 0)
        with torch.cuda.device(self.device):
            try: 
                self.load_model(kwargs["model"], kwargs["controlnet"])
            except KeyError:
                print("Model name and controlnet model must be specified")
                raise

        self.generator = torch.Generator(device=self.device)
        self.cpu_generator = torch.Generator(device="cpu")
        self.generator.manual_seed(42)
        self.generator_init_state = self.generator.get_state()
        self.cpu_generator_init_state = self.cpu_generator.get_state()


    def compile_model(self):

        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
        self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead", fullgraph=True)
        self.model.vae = torch.compile(self.model.vae, mode="reduce-overhead", fullgraph=True)

        kwarg_inputs = dict(
            prompt="warmup",
            image=[Image.new("RGB", (768, 768))],
            control_image=[Image.new("RGB", (768, 768))],
        )
        return self.model(**kwarg_inputs).images[0]

    def load_model(self, model_name, controlnet_model="lllyasviel/sd-controlnet-canny"):
        #self.controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float32)
        # self.model = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model_name, 
        #                                                             controlnet=self.controlnet, torch_dtype=torch.float16)
        #self.model = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main", revision="fb9c5d")
        # self.model = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
        #         pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
        #         safety_checker=None
        #     )

        controlnet_canny = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
        ).to(self.device)

        self.canny_torch = SobelOperator(device=self.device)

        model_id = "SimianLuo/LCM_Dreamshaper_v7"

        self.model = LatentConsistencyModelPipeline_controlnet.from_pretrained(
            model_id,
            safety_checker=None,
            controlnet=controlnet_canny,
            torch_dtype=torch.float16,
            scheduler=None,
        )

        self.model.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=torch.float16, use_safetensors=True
        )

        # self.model = LatentConsistencyModelPipeline_reference(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor)
        # self.model.safety_checker = None
        self.model.to(torch.device(self.device))
        self.model.unet.to(memory_format=torch.channels_last)
        return self.model
    

    def infer(
        self,
        img,
        prompt,
        negative_prompt=["deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"],
        height=360,
        width=640,
        strength=0.4,
        num_of_infer_steps=20,  
        guidance_scale=7.5,
        seed=42,
        warmup=False,
        verbose=False,
        ref_frame=False,
        ref=False,
        set_ref=False,
        style_fidelity = 1,
        controlnet_scale = 1,
        controlnet=True
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
        # if image width and height don't match


        # center crop image to match desired aspect ratio
        if img.width / img.height > width / height:
            # crop width
            new_width = img.height * (width / height)
            left = (img.width - new_width) / 2
            top = 0
            right = (img.width + new_width) / 2
            bottom = img.height
        else:
            # crop height
            new_height = img.width * (height / width)
            left = 0
            top = (img.height - new_height) / 2
            right = img.width
            bottom = (img.height + new_height) / 2
        img = img.crop((left, top, right, bottom))
        img = img.resize((width, height), resample=Image.Resampling.LANCZOS)
        #ref_frame = ref_frame.resize((width, height), resample=Image.Resampling.LANCZOS)
        assert guidance_scale > 1.0
        self.guidance_scale = guidance_scale
        #canny_image = np.array(img)
        canny_image = self.canny_torch(img, 0.31, 0.8)
        self.generator.set_state(self.generator_init_state)
        self.generator.manual_seed(seed) 
        np.random.seed(seed)

        kwarg_inputs = dict(
            prompt=prompt,
            #negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_of_infer_steps,
            image=img,
            control_image=canny_image,
            #ref_image=ref_frame,
            #style_fidelity=style_fidelity,
            controlnet_conditioning_scale=style_fidelity,
            generator=self.generator,
            strength=strength
        )
        
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)

        torch.manual_seed(seed).set_state(self.cpu_generator_init_state)
        
        return self.model(**kwarg_inputs).images[0]
