FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG GROUP_ID

WORKDIR /workspace

RUN apt-get update 
RUN apt-get install -y build-essential
RUN apt-get install -y software-properties-common wget curl git
RUN apt-get clean

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch torchvision torchaudio
RUN python3 -m pip install timm requests wandb transformers black matplotlib

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENTRYPOINT ["/bin/bash"]
