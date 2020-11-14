FROM nvcr.io/nvidia/pytorch:20.10-py3

ENV SHELL=/bin/bash
ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
USER root

RUN rm -rf /opt/conda/ && \
    apt update && apt install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    gnupg \
    tmux \
    sudo \
    ssh \
    nano \
    mysql-client \
    libpq-dev \
    git \
    vim \
    wget \
    curl \
    sqlite \
    sqlite3 \
    libgl1-mesa-glx \
    software-properties-common


RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.8 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 

RUN apt install -y python3-pip && \
    pip3 install --upgrade pip


RUN pip3 install matplotlib numpy pandas pillow scikit-image scikit-learn scipy seaborn tqdm albumentations opencv-python pytorch_toolbelt
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


WORKDIR /workspace/ai_haste
