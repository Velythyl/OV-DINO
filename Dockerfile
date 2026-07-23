# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libglib2.0-0 \
        libgl1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip

WORKDIR /opt/ovdino

# PyTorch must match both the CUDA base image and the project's supported version.
RUN python3 -m pip install \
        torch==1.13.1+cu116 \
        torchvision==0.14.1+cu116 \
        torchaudio==0.13.1 \
        --extra-index-url https://download.pytorch.org/whl/cu116

COPY ovdino/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

COPY ovdino /opt/ovdino

# Build the project's CUDA extensions while the CUDA toolkit is available.
RUN FORCE_CUDA=1 python3 -m pip install -e detectron2-717ab9 \
    && FORCE_CUDA=1 python3 -m pip install -e . \
    && TRANSFORMERS_OFFLINE=0 python3 -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased')"

COPY docker/entrypoint.sh /usr/local/bin/ovdino-demo
RUN chmod +x /usr/local/bin/ovdino-demo \
    && mkdir -p /weights /workspace

WORKDIR /opt/ovdino
EXPOSE 7860

ENTRYPOINT ["ovdino-demo"]
