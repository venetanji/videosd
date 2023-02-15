import argparse
import gc
from pathlib import Path
import os
from typing import List

import tensorrt as trt
import torch
from polygraphy import cuda
from models import CLIP, VAE, UNet
import onnx
from utilities import TRT_LOGGER, Engine

def parseArgs():
    "Parse command line arguments"

    parser = argparse.ArgumentParser(description="Options for VideoSDTrack accelerated engine")
    parser.add_argument(
        "--height",
        type=int,
        default=384,
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
        "--denoising-steps", type=int, default=20, help="Number of denoising steps"
    )
    parser.add_argument(
        "--denoising-prec",
        type=str,
        default="fp16",
        choices=["fp32", "fp16"],
        help="Denoiser model precision",
    )

    # ONNX export
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=16,
        choices=range(7, 18),
        help="Select ONNX opset version to target for exported models",
    )
    parser.add_argument(
        "--onnx-dir", default="/onnx", help="Output directory for ONNX export"
    )
    parser.add_argument(
        "--onnx-minimal-optimization",
        action="store_true",
        help="Restrict ONNX optimization to const folding and shape inference",
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
        "--build-dynamic-shape",
        action="store_true",
        help="Build TensorRT engines with dynamic image shapes",
    )
    parser.add_argument(
        "--enable-preview-features",
        action="store_true",
        help="Disable TensorRT preview features",
    )

    # TensorRT inference
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace API access token for downloading model checkpoints",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )

    return parser.parse_args()

 
def compile_trt(
    model,
    img_height,
    img_width
):

    args = parseArgs()
    print("[I] Building TensorRT engine with args:", args)

    print("[I] Model: ", model)
    
    engines_dir = Path(args.engine_dir)
    engines_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model_engine_dir = engines_dir / model
    model_engine_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    verbose = args.verbose
    hf_token = args.hf_token
    max_batch_size = 1
    models = {
            "unet_fp16": UNet(
                model_path=model,
                hf_token=hf_token,
                fp16=True,
                device=device,
                verbose=verbose,
                max_batch_size=max_batch_size,
            ),
            "clip": CLIP(
                hf_token=hf_token,
                device=device,
                verbose=verbose,
                max_batch_size=max_batch_size,
            ),
            "vae": VAE(
                hf_token=hf_token,
                device=device,
                verbose=verbose,
                max_batch_size=max_batch_size,
            )
        }
    
    # Register TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    for model, trtmodel in models.items():
        onnx_path = onnx_dir / f"{model}.onnx"
        onnx_opt_path = onnx_dir / f"{model}.opt.onnx"
        export_onnx(onnx_path, trtmodel, img_width, img_height)
        optimize_onnx(onnx_path, trtmodel, onnx_opt_path)
    
    # Clear CUDA cache to avoid OOM errors during engine build
    torch.cuda.empty_cache()

    for model, trtmodel in models.items():  
        profile = trtmodel.get_input_profile(
                        1,
                        img_height,
                        img_width,
                        static_batch=True,
                        static_shape=not args.build_dynamic_shape)
        compile_engine(model, profile, model_engine_dir, onnx_dir)


def export_onnx(onnx_path,trtmodel,img_width,img_height):
    if not onnx_path.exists():
        print(f"Exporting model: {onnx_path}")
        model = trtmodel.get_model()
        with torch.inference_mode(), torch.autocast("cuda"):
            inputs = trtmodel.get_sample_input(
                1, img_height, img_width
            )
            print("Starting export of ONNX model")
            torch.onnx.export(
                model,
                inputs,
                onnx_path,
                export_params=True,
                opset_version=args.onnx_opset,
                do_constant_folding=True,
                input_names=trtmodel.get_input_names(),
                output_names=trtmodel.get_output_names(),
                dynamic_axes=trtmodel.get_dynamic_axes(),
            )
            print("Finished export of ONNX model")
        del model,inputs
    else:
        print(f"Found cached model: {onnx_path}")

def optimize_onnx(onnx_path, trtmodel, onnx_opt_path ):
    if not os.path.exists(onnx_opt_path):
        print(f"Generating optimizing model: {onnx_opt_path}")
        onnx_opt_graph = trtmodel.optimize(
            onnx.load(onnx_path),
            minimal_optimization=args.onnx_minimal_optimization,
        )
        onnx.save(onnx_opt_graph, onnx_opt_path)
        del onnx_opt_graph
    else:
        print(f"Found cached optimized model: {onnx_opt_path}")

def compile_engine(model, profile, model_engine_dir, onnx_dir):
    engine = Engine(model, model_engine_dir)
    engine_path = model_engine_dir / f"{model}.plan"
    onnx_opt_path = onnx_dir / f"{model}.opt.onnx"

    if not engine_path.exists():
        print("Building the TRT engine...")
        engine.build(
            str(onnx_opt_path),
            fp16=True,
            input_profile=profile,
            enable_preview=args.enable_preview_features,
        )
    del engine

if __name__ == "__main__":
    print("Building engine...")
    args = parseArgs()
    compile_trt(
        model=args.model_path,
        img_height=args.height,
        img_width=args.width
    )

