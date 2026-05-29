"""
一次性模型预下载脚本。

用法：
  挂载 Network Volume 到 /runpod-volume 的 Pod 上跑一次：
    python3 download_models.py

下载完后，Serverless endpoint 启动时就不用再下模型 = 秒级就绪。
"""
import os
import subprocess
import sys
from pathlib import Path

# 确认 Network Volume 已挂载
VOLUME = Path("/runpod-volume")
if not VOLUME.exists():
    print(f"[FATAL] {VOLUME} 不存在 — 这个脚本必须在挂了 Network Volume 的 Pod 上运行")
    sys.exit(1)

# 同步 env (handler 也用这套)
os.environ["HF_HOME"] = "/runpod-volume/cache/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/runpod-volume/cache/hf"
os.environ["PADDLE_PDX_CACHE_HOME"] = "/runpod-volume/cache/paddlex"
os.environ["MINERU_MODEL_SOURCE"] = "local"
os.environ["MINERU_MODELS_DIR"] = "/runpod-volume/cache/mineru"

# Paddle 兜底 symlink
home_paddlex = Path.home() / ".paddlex"
volume_paddlex = Path("/runpod-volume/cache/paddlex")
volume_paddlex.mkdir(parents=True, exist_ok=True)
if not home_paddlex.exists():
    home_paddlex.symlink_to(volume_paddlex)


def step(n, total, label):
    print(f"\n{'='*60}\n[{n}/{total}] {label}\n{'='*60}", flush=True)


def download_whisper():
    step(1, 3, "faster-whisper large-v3 (~3 GB)")
    from faster_whisper import WhisperModel
    # CPU 加载即可触发下载，省 GPU
    WhisperModel("large-v3", device="cpu", compute_type="int8")
    print("  ✓ faster-whisper 模型已下载")


def download_paddleocr():
    step(2, 3, "PaddleOCR PP-StructureV3 多个子模型 (~500 MB)")
    from paddleocr import PPStructureV3
    # 实例化触发所有子模型下载
    PPStructureV3(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        use_chart_recognition=False,
        use_formula_recognition=False,
        use_seal_recognition=False,
    )
    print("  ✓ PaddleOCR 模型已下载")


def download_mineru():
    step(3, 3, "MinerU 全部模型 (~5-10 GB)")
    # mineru 提供 CLI 工具批量下载
    result = subprocess.run(
        ["mineru-models-download", "-s", "huggingface", "-m", "all"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("  [warn] mineru-models-download 失败，尝试备用 source...")
        subprocess.run(
            ["mineru-models-download", "-s", "modelscope", "-m", "all"],
            check=True,
        )
    print("  ✓ MinerU 模型已下载")


if __name__ == "__main__":
    download_whisper()
    download_paddleocr()
    download_mineru()
    # 检查容量
    subprocess.run(["du", "-sh", "/runpod-volume/cache"], check=False)
    print("\n✅ 全部模型已落地 Network Volume，可以销毁这个 Pod 了。")
