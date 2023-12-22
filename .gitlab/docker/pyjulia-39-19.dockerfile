
# syntax=docker/dockerfile:1
FROM julia:1.9-bullseye AS jlImage

FROM python:3.9-slim-bullseye
WORKDIR /

COPY --from=jlImage /usr/local/julia /usr/local/julia

# Set julia environment variables (see: https://julialang.org/downloads/ and https://github.com/docker-library/julia/)
ENV JULIA_PATH=/usr/local/julia
ENV PATH=$JULIA_PATH/bin:$PATH
ENV JULIA_VERSION=1.9.3
ENV PYCALL_JL_RUNTIME_PYTHON=$PYTHON
ENV PATH=$PATH:/root/.local/bin

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
    apt-get install -y --no-install-recommends git make curl && \
    python -m pip install --upgrade "pip<22" && \
    python -m pip install pipx && \
    export PATH=$PATH:/root/.local/bin && \
    python -m pipx ensurepath


# Install poetry (and pytorch, because we only need the cpu version)
RUN pipx install poetry==1.6.1 && \
    pip install torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

# Copy necessary files for poetry and pip (only for installation)
COPY pyproject.toml poetry.lock LICENSE README.rst ./
COPY eta_utility ./eta_utility

# Export all pinned dependencies to requirements.txt for pip installation
RUN poetry export --extras develop --without-hashes -o requirements.txt && \
    # Remove nvidia and torch from requirements.txt, pytorch is installed above and nvidia packages (cuda) are not needed
    sed -i '/^nvidia-/d' ./requirements.txt && \
    sed -i '/torch==/d' ./requirements.txt && \
    pip install -r requirements.txt --no-cache-dir && \
    pip install . --no-deps

# Install julia
RUN \
    pip install julia --no-cache-dir && \
    julia -e 'ENV["PYTHON"] = Sys.which("python"); using Pkg; Pkg.add("PyCall"); Pkg.build("PyCall"); Pkg.add("JuliaFormatter")' && \
    install-julia

# Remove unnecessary files
RUN rm -rf ${APP_PATH} && \
    mkdir ${APP_PATH}
