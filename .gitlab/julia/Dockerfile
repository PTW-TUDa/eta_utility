# creates an image with Tensorflow ist already installed
FROM julia:1.8-bullseye

ENV APP_PATH=/usr/src/app
WORKDIR $APP_PATH

# Install python
RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-virtualenv \
    python3-venv

RUN rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends git

RUN python3 -m venv venv

RUN \
    . venv/bin/activate && \
    python -m pip install --upgrade pip && \
    pip install --upgrade "setuptools<67" --no-cache-dir && \
    pip install julia --no-cache-dir && \
    pip install eta_utility[develop] --no-cache-dir && \
    pip uninstall eta_utility --yes && \
    julia -e 'ENV["PYTHON"] = Sys.which("python"); using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall")'
