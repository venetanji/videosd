FROM nvcr.io/nvidia/tritonserver:23.09-pyt-python-py3
WORKDIR /workspace/diffusert
COPY diffusert/requirements.txt /workspace/diffusert/
RUN apt-get update && apt-get install libgl1 -y
RUN pip install -r /workspace/diffusert/requirements.txt
RUN pip install -v -U git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast
