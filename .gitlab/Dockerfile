# creates an image with Tensorflow ist already installed
FROM julia:1.8-bullseye

ENV APP_PATH=/usr/src/app
WORKDIR $APP_PATH

# Install python
RUN \
  apt-get update && \
  apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-virtualenv \
    python3-venv

RUN rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends git