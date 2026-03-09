# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=10
ENV FLASH_ATTENTION_DISABLE_SM80=TRUE
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . /workspace/repo

# Set up environment (clones and builds FA3)
WORKDIR /workspace/repo
RUN bash scripts/setup_environment.sh

# Default command: run full reproduction in quick mode
CMD ["python", "reproduce.py", "--skip-setup", "--quick"]
