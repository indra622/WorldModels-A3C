FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    build-essential \
    python3-dev \
    cmake \
    ffmpeg \
    xvfb \
    libgl1-mesa-glx \
    libglu1-mesa \
    libosmesa6 \
    libglfw3 \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

CMD ["bash"]
