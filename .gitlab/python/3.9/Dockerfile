FROM python:3.9-slim-bullseye

ENV APP_PATH=/usr/src/app
WORKDIR $APP_PATH

RUN \
    apt-get -y update && \
    apt-get -y install --no-install-recommends git make curl

# setuptools<67 because higher versions are incompatible with gym==0.21 (required by stable_baselines)
RUN \
    python -m pip install --upgrade pip && \
    pip install --upgrade "setuptools<67" --no-cache-dir && \
    pip install eta_utility[develop] --no-cache-dir && \
    pip uninstall eta_utility --yes
