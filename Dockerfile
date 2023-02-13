
FROM nvcr.io/nvidia/tensorrt:23.01-py3
ENV CUDA_MODULE_LOADING=LAZY
WORKDIR /workspace/diffusert
COPY diffusert/requirements.txt /workspace/diffusert/
RUN pip install -r /workspace/diffusert/requirements.txt
