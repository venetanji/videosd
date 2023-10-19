FROM nvcr.io/nvidia/pytorch:23.08-py3
WORKDIR /workspace
RUN git clone https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /workspace/ComfyUI
RUN git clone https://github.com/FizzleDorf/AIT custom_nodes/AIT
RUN pip install -r requirements.txt
COPY extra_model_paths.yaml extra_model_paths.yaml


