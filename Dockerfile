# Multi-stage Dockerfile for optimal image size and security
# Build stage
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt /tmp/

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Runtime stage
FROM python:3.11-slim-bookworm AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create application directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p storage logs models .cache && \
    chown -R botuser:botuser /app

# Copy application code
COPY --chown=botuser:botuser . .

# Download models (optional - can be done at runtime)
# RUN python -c "import whisper; whisper.load_model('base', download_root='models/whisper')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8443/health')" || exit 1

# Switch to non-root user
USER botuser

# Expose webhook port (if using webhook mode)
EXPOSE 8443

# Run the bot
CMD ["python", "main.py"]

# =============================================
# Development stage (optional)
# =============================================
FROM runtime AS development

# Switch to root for installing dev tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN /opt/venv/bin/pip install --no-cache-dir \
    ipython \
    ipdb \
    pytest \
    pytest-asyncio \
    black \
    flake8

# Switch back to non-root user
USER botuser

# =============================================
# GPU stage (for NVIDIA GPUs)
# =============================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS gpu-runtime

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install with CUDA support
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Create non-root user
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create application directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p storage logs models .cache && \
    chown -R botuser:botuser /app

# Copy application code
COPY --chown=botuser:botuser . .

# Switch to non-root user
USER botuser

# Run the bot
CMD ["python", "main.py"]