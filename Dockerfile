FROM nvcr.io/nvidia/tritonserver:23.09-pyt-python-py3
WORKDIR /workspace/diffusert
RUN apt-get update && apt-get install libgl1 -y
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
COPY diffusert/requirements.txt /workspace/diffusert/
RUN pip install -r /workspace/diffusert/requirements.txt
