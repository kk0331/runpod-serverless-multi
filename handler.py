"""
RunPod Serverless handler — 三合一服务路由

输入格式 (job.input):
  {
    "task": "asr" | "ocr" | "mineru",
    "url":  "<HTTPS URL to file>"   # 二选一
    "data": "<base64-encoded file>" # 二选一
    "language": "zh|en|..."         # 仅 asr 可选，省略=自动检测
    "options": {...}                # 任务特定参数
  }

输出格式:
  asr     -> {"language": "zh", "duration": 1234.5, "text": "...", "segments": [...]}
  ocr     -> {"markdown": "...", "page_count": 42}
  mineru  -> {"markdown": "...", "page_count": 42}
  错误     -> {"error": "...", "traceback": "..."}
"""
import os
import sys
import base64
import glob
import re
import shutil
import tempfile
import traceback
import urllib.request
import subprocess
from pathlib import Path

import runpod


# --- 启动时一次性准备 ---
# Paddle 的某些子模型不认 PADDLE_PDX_CACHE_HOME，仍会写 ~/.paddlex
# 用 symlink 兜底：把 ~/.paddlex 指向 Network Volume
def _setup_paddle_cache():
    home_paddlex = Path.home() / ".paddlex"
    volume_paddlex = Path("/runpod-volume/cache/paddlex")
    volume_paddlex.mkdir(parents=True, exist_ok=True)
    if home_paddlex.is_symlink() or home_paddlex.exists():
        return
    home_paddlex.parent.mkdir(parents=True, exist_ok=True)
    try:
        home_paddlex.symlink_to(volume_paddlex)
    except OSError as e:
        print(f"[warn] symlink ~/.paddlex failed: {e}", file=sys.stderr)


_setup_paddle_cache()

# --- 模型懒加载缓存 ---
_models = {}


def _get_whisper():
    if "whisper" in _models:
        return _models["whisper"]
    from faster_whisper import WhisperModel
    print("[load] faster-whisper large-v3 (cuda, float16)", flush=True)
    _models["whisper"] = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
    )
    return _models["whisper"]


def _get_paddleocr():
    if "ocr" in _models:
        return _models["ocr"]
    from paddleocr import PPStructureV3
    print("[load] PaddleOCR PP-StructureV3", flush=True)
    # 沿用 KK 在 convert_auto.py 中的参数配置 (中文古籍/史料友好)
    _models["ocr"] = PPStructureV3(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        text_det_limit_side_len=1536,
        text_det_limit_type="min",
        text_det_thresh=0.2,
        text_rec_score_thresh=0.0,
        use_chart_recognition=False,
        use_formula_recognition=False,
        use_seal_recognition=False,
    )
    return _models["ocr"]


# --- I/O 辅助 ---
def _materialize_input(job_input, suffix=""):
    """把 input 里的 url 或 base64 data 落到本地临时文件，返回路径。"""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if "url" in job_input:
        print(f"[io] downloading {job_input['url'][:80]}...", flush=True)
        urllib.request.urlretrieve(job_input["url"], path)
    elif "data" in job_input:
        with open(path, "wb") as f:
            f.write(base64.b64decode(job_input["data"]))
    else:
        raise ValueError("input must contain either 'url' or 'data'")
    return path


# --- 任务实现 ---
def task_asr(job_input):
    audio_path = _materialize_input(job_input, suffix=".audio")
    try:
        model = _get_whisper()
        segments_iter, info = model.transcribe(
            audio_path,
            language=job_input.get("language"),
            beam_size=job_input.get("beam_size", 5),
            vad_filter=job_input.get("vad_filter", True),
            initial_prompt=job_input.get("initial_prompt"),
        )
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_iter]
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "text": "".join(s["text"] for s in segments),
            "segments": segments,
        }
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


def task_ocr(job_input):
    pdf_path = _materialize_input(job_input, suffix=".pdf")
    out_dir = tempfile.mkdtemp(prefix="ocr_")
    try:
        pipe = _get_paddleocr()
        for res in pipe.predict(pdf_path):
            res.save_to_markdown(save_path=out_dir)

        md_files = glob.glob(os.path.join(out_dir, "*.md"))

        def page_key(p):
            m = re.search(r"_(\d+)\.md$", p)
            return int(m.group(1)) if m else 0

        md_files.sort(key=page_key)

        pages = []
        for f in md_files:
            with open(f, encoding="utf-8") as fh:
                content = fh.read().strip()
            if content:
                pages.append(content)

        return {
            "markdown": "\n\n---\n\n".join(pages),
            "page_count": len(pages),
        }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        try:
            os.remove(pdf_path)
        except OSError:
            pass


def task_mineru(job_input):
    pdf_path = _materialize_input(job_input, suffix=".pdf")
    out_dir = tempfile.mkdtemp(prefix="mineru_")
    try:
        backend = job_input.get("backend", "pipeline")  # pipeline | vlm-transformers
        cmd = ["mineru", "-p", pdf_path, "-o", out_dir, "-b", backend]
        lang = job_input.get("language")
        if lang:
            cmd.extend(["-l", lang])
        print(f"[mineru] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(
                f"mineru failed (exit {proc.returncode})\n"
                f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
            )
        md_files = glob.glob(os.path.join(out_dir, "**", "*.md"), recursive=True)
        if not md_files:
            raise RuntimeError(f"no .md output; stdout={proc.stdout[-1000:]}")
        with open(md_files[0], encoding="utf-8") as f:
            return {"markdown": f.read(), "output_file": os.path.basename(md_files[0])}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        try:
            os.remove(pdf_path)
        except OSError:
            pass


# --- RunPod handler ---
TASK_MAP = {"asr": task_asr, "ocr": task_ocr, "mineru": task_mineru}


def handler(job):
    job_input = job.get("input", {}) or {}
    task = job_input.get("task")
    if task not in TASK_MAP:
        return {"error": f"unknown task: {task!r}. valid: {list(TASK_MAP)}"}
    try:
        return TASK_MAP[task](job_input)
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
