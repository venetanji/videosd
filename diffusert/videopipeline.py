import numpy as np
import time
import torch
import cv2
from sfast.compilers.stable_diffusion_pipeline_compiler import compile, CompilationConfig
#from diffusers import DiffusionPipeline, ControlNetModel
import packaging.version
#from .lcm.lcm_pipeline import LatentConsistencyModelPipeline
#from lcm.lcm_i2i_pipeline import LatentConsistencyModelImg2ImgPipeline
from lcm.lcm_reference_pipeline import LatentConsistencyModelPipeline_reference, LCMScheduler_X
from diffusers import AutoencoderKL, UNet2DConditionModel
#from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor


from PIL import Image

if packaging.version.parse(torch.__version__) >= packaging.version.parse('1.12.0'):
    torch.backends.cuda.matmul.allow_tf32 = True

class VideoSDPipeline:
    """
    Application showcasing the acceleration of Stable Diffusion Txt2Img v1.4, v1.5, v2.0, v2.0-base, v2.1, v2.1-base pipeline using NVidia TensorRT w/ Plugins.
    """
    def __init__(
        self,
        *args, **kwargs
    ):
        self.device = kwargs.get("device", "cuda")
        with torch.cuda.device(self.device):
            try: 
                self.load_model(kwargs["model"], kwargs["controlnet"])
            except KeyError:
                print("Model name and controlnet model must be specified")
                raise

        self.generator = torch.Generator(device=self.device)
        self.generator_init_state = self.generator.get_state()


    def compile_model(self):
        torch.set_grad_enabled(False)
        #torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.model.text_encoder = torch.compile(self.model.text_encoder, mode='max-autotune')
        self.model.tokenizer = torch.compile(self.model.tokenizer, mode='max-autotune')
        self.model.unet = torch.compile(self.model.unet, mode='max-autotune')
        self.model.vae = torch.compile(self.model.vae, mode='max-autotune')

        # config = CompilationConfig.Default()
        # config.enable_xformers = True
        # config.enable_triton = True
        # config.enable_cuda_graph = True
        # self.compiled_model = compile(self.model, config)

        # # create a dummy noisy image 640x360
        image = Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))

        canny_image = self.get_canny_filter(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8), low_threshold=100, high_threshold=200)
        
        kwarg_inputs = dict(
            prompt='a dog',
            height=360,
            width=640,
            num_inference_steps=4,
            strength=0.4,
            image=image,

            control_image=canny_image,
            num_images_per_prompt=1
        
            )
        
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
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
        model_id = "SimianLuo/LCM_Dreamshaper_v7"

        # Initalize Diffusers Model:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", device_map=None, low_cpu_mem_usage=False, local_files_only=True)
        #safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")
        scheduler = LCMScheduler_X(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")

        self.model = LatentConsistencyModelPipeline_reference(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=feature_extractor)
        self.model.safety_checker = None
        self.model.to(torch.device(self.device))
        return self.model
    
    def get_canny_filter(self, image, low_threshold=100, high_threshold=200):
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

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
        negative_prompt=["deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"],
        image_height=360,
        image_width=640,
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
        img = img.resize((640, 360), resample=Image.Resampling.LANCZOS)
        ref_frame = ref_frame.resize((640, 360), resample=Image.Resampling.LANCZOS)
        assert guidance_scale > 1.0
        self.guidance_scale = guidance_scale
        #canny_image = np.array(img)
        #canny_image = self.get_canny_filter(canny_image)
        self.generator.manual_seed(seed)  
        self.generator.set_state(self.generator_init_state)
        np.random.seed(seed)

        kwarg_inputs = dict(
            prompt=prompt,
            #negative_prompt=negative_prompt,
            height=image_height,
            width=image_width,
            num_inference_steps=num_of_infer_steps,
            image=img,
            ref_image=ref_frame,
            style_fidelity=style_fidelity,
            generator=self.generator,
            strength=strength
        )
        
        torch.cuda.set_device(self.device)
        torch.cuda.synchronize(self.device)
        torch.cuda.manual_seed(seed)

        if hasattr(self, "compiled_model"):
            return self.compiled_model(**kwarg_inputs).images[0]
        else:
            return self.model(**kwarg_inputs).images[0]
