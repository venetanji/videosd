import numpy as np
import torch

from lcm.lcm_controlnet import LatentConsistencyModelPipeline_controlnet
from lcm.canny_gpu import SobelOperator 
from diffusers import AutoencoderTiny, ControlNetModel
import ray
from PIL import Image


@ray.remote(num_gpus=1, num_cpus=4)
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

    def load_model(self, model_name, controlnet_model="lllyasviel/control_v11p_sd15_canny"):

        controlnet_canny = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch.float16
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
        self.model.to(torch.device(self.device))
        self.model.unet.to(memory_format=torch.channels_last)
        return self.model
    

    def infer(
        self,
        img,
        prompt=["pixar, cg"],
        height=360,
        width=640,
        strength=0.4,
        steps=20,  
        guidance_scale=7.5,
        ref=False,
        style_fidelity=0.0,
        controlnet=False,
        seed=42,
        controlnet_scale = 1,
    ):

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

        canny_image = self.canny_torch(img, 0.11, 0.8)
        self.generator.set_state(self.generator_init_state)
        self.generator.manual_seed(seed) 
        np.random.seed(seed)

        kwarg_inputs = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            image=img,
            control_image=canny_image,
            controlnet_conditioning_scale=controlnet_scale,
            generator=self.generator,
            strength=strength
        )

        torch.manual_seed(seed).set_state(self.cpu_generator_init_state)
        
        return self.model(**kwarg_inputs).images[0]
