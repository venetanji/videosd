
FROM nvcr.io/nvidia/tensorrt:23.06-py3
ENV CUDA_MODULE_LOADING=LAZY
WORKDIR /workspace/diffusert
COPY diffusert/requirements.txt /workspace/diffusert/
RUN apt-get update && apt-get install libgl1 -y
RUN pip install -r /workspace/diffusert/requirements.txt
