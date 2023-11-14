# syntax=docker/dockerfile:1
FROM python:3.8-slim-bullseye

# Set workspace
ENV APP_PATH=/usr/src/app
WORKDIR $APP_PATH

# Test
RUN python --version

# Install additional dependencies for pipelines
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends git make curl; \
    rm -rf /var/lib/apt/lists/*;

# Install poetry (and pytorch, because we only need the cpu version)
RUN pip install "poetry==1.6.1" --no-cache-dir && \
    pip install torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

# Copy necessary files for poetry and pip (only for installation)
COPY pyproject.toml poetry.lock LICENSE README.rst ./

# Export all pinned dependencies to requirements.txt for pip installation
RUN poetry export --extras develop --without-hashes -o requirements.txt && \
    # Remove nvidia and torch from requirements.txt, pytorch is installed above and nvidia packages (cuda) are not needed
    sed -i '/^nvidia-/d' ./requirements.txt && \
    sed -i '/torch==/d' ./requirements.txt && \
    pip install -r requirements.txt --no-cache-dir

# Remove unnecessary files
RUN rm -rf ${APP_PATH} && \
    mkdir ${APP_PATH}
