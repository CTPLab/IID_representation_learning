FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ARG APT_INSTALL="apt-get install -y"
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN $APT_INSTALL build-essential software-properties-common ca-certificates \
                 nano wget git zlib1g-dev nasm cmake ffmpeg libsm6 libxext6

RUN pip install wilds matplotlib opencv-python ninja
WORKDIR /root