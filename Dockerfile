# RunPod Serverless: 三合一 endpoint
# 服务: ASR (faster-whisper) + OCR (PaddleOCR PP-StructureV3) + MinerU PDF→Markdown
#
# 基础镜像选 MinerU 官方推荐 (CUDA 12.9 + Python 3.11 + torch)
# 这是 MinerU 3.x 测试过的环境，最大化"保证能跑通"的概率
FROM vllm/vllm-openai:v0.21.0-cu129

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# 重要：vllm 镜像默认 ENTRYPOINT 是启动 vllm server，必须清掉
ENTRYPOINT []

# --- 系统依赖 ---
# fonts-noto-cjk: MinerU 中文渲染必需
# ffmpeg: faster-whisper 音频处理
# libgl1, libsndfile1: 常见 ML 依赖
# poppler-utils: PDF 工具
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-core fonts-noto-cjk fontconfig \
        libgl1 libsndfile1 \
        ffmpeg poppler-utils \
        ca-certificates curl \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# --- Python 依赖 ---
# 分层 install: 失败时可单层重试，并改善 docker layer 缓存

# 1) PaddlePaddle GPU (cu126 wheel: ABI 兼容 CUDA 12.9 主机驱动)
RUN python3 -m pip install \
        paddlepaddle-gpu==3.3.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 2) PaddleOCR + doc-parser extra (含 PP-StructureV3)
RUN python3 -m pip install \
        "paddleocr[doc-parser]==3.6.0"

# 3) faster-whisper (CTranslate2 4.x + CUDA 12 / cuDNN 9)
RUN python3 -m pip install \
        faster-whisper==1.2.1 \
        nvidia-cublas-cu12 \
        nvidia-cudnn-cu12

# 4) MinerU core (vlm + pipeline + gradio，不含 vllm 集成层；vllm 已在基础镜像)
RUN python3 -m pip install \
        "mineru[core]>=3.2.1"

# 5) 工具库 + RunPod SDK
RUN python3 -m pip install \
        PyMuPDF \
        runpod \
        boto3

# --- 缓存重定向到 Network Volume ---
# RunPod Serverless 把 Network Volume 自动挂载到 /runpod-volume
# 把 HF / PaddleX / MinerU 的模型缓存全部指过去 = 模型只下载一次，所有 worker 共享
ENV HF_HOME=/runpod-volume/cache/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/cache/hf \
    PADDLE_PDX_CACHE_HOME=/runpod-volume/cache/paddlex \
    MINERU_MODEL_SOURCE=local \
    MINERU_MODELS_DIR=/runpod-volume/cache/mineru

# --- handler ---
WORKDIR /workspace
COPY handler.py /workspace/handler.py
COPY download_models.py /workspace/download_models.py

CMD ["python3", "-u", "/workspace/handler.py"]
