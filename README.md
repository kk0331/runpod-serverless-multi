# RunPod Serverless 三合一 Endpoint

一个 Serverless endpoint，三个服务，按需启动：

- **`asr`** — faster-whisper large-v3 多语言语音转文字（含中文）
- **`ocr`** — PaddleOCR PP-StructureV3 中文 PDF → Markdown（含古籍、影印件、竖排）
- **`mineru`** — MinerU 3.x 通用 PDF → Markdown（含公式、表格、英文学术论文）

---

## 架构

```
┌─────────────────────┐    submit {task, url|data}    ┌──────────────────────────┐
│  Mac / 其他客户端    │ ────────────────────────────> │  RunPod Serverless        │
│  au-transcribe etc. │                                │  Endpoint                 │
└─────────────────────┘ <──────────────────────────── │  ├─ Docker 镜像 (GHCR)    │
                            {markdown | text}         │  ├─ Network Volume        │
                                                      │  │   /runpod-volume/cache │
                                                      │  │   ├─ hf/      whisper │
                                                      │  │   ├─ paddlex/ ocr     │
                                                      │  │   └─ mineru/  mineru  │
                                                      │  └─ A40 48GB Flex worker │
                                                      └──────────────────────────┘
```

**关键设计**：
- 镜像里只有代码 + 依赖（~15 GB），**模型全在 Network Volume**
- 首次起 Pod 跑 `download_models.py` 把所有模型预下到 Volume
- 之后每个 Serverless worker 启动 = 拉镜像（缓存后秒级）+ 挂 Volume + load handler ≈ **30 秒–3 分钟**
- Idle Timeout 5 分钟：连续提交不冷启动

---

## 一次性部署流程

### 1. 把代码推上 GitHub

```bash
cd /Users/kk2/runpod-serverless-multi
git init
git add .
git commit -m "Initial: three-in-one Serverless endpoint"
# 在 GitHub 建 public repo (我会用 CDP 帮你建)，然后：
git remote add origin https://github.com/<你的GH用户名>/runpod-serverless-multi.git
git branch -M main
git push -u origin main
```

### 2. 在 RunPod 起一个临时 CPU Pod 来 build + push 镜像

- GPU 选最便宜的（CPU 也行，build 不需要 GPU）
- 镜像选 `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`（自带 docker）
- 挂载 Network Volume（重要：build 完后顺便预下模型）

在 Pod 里：

```bash
# 装 docker buildx (Pod 通常已有)
docker buildx version

# 登录 GHCR (用 GitHub Personal Access Token，需 write:packages 权限)
# Token 创建: https://github.com/settings/tokens/new (scopes: write:packages, read:packages)
echo "<GHCR_PAT>" | docker login ghcr.io -u <你的GH用户名> --password-stdin

# 克隆 repo
git clone https://github.com/<你的GH用户名>/runpod-serverless-multi.git
cd runpod-serverless-multi

# 构建并推送 (~10–20 分钟，含下载基础镜像)
GH_USER=<你的GH用户名>
IMG=ghcr.io/${GH_USER}/runpod-serverless-multi:latest
docker build -t $IMG .
docker push $IMG
```

### 3. 在同一个 Pod 上预下载模型到 Network Volume

```bash
# 假定 Network Volume 挂在 /runpod-volume
python3 download_models.py
# 完成后看占用
du -sh /runpod-volume/cache/*
```

下完销毁这个 Pod。

### 4. 在 RunPod Console 创建 Serverless Endpoint

- Container Image: `ghcr.io/<你的GH用户名>/runpod-serverless-multi:latest`
- Network Volume: 选刚才用的那个
- GPU Type: **A40 48GB**
- Active Workers: **0**（Flex 模式，省钱）
- Max Workers: **2**（兜底重试）
- Idle Timeout: **300s**（5 分钟保温）
- Container Disk: 20 GB（够装镜像）

---

## 调用示例

```python
import requests, base64

API = "https://api.runpod.ai/v2/<endpoint-id>/runsync"
HEADERS = {"Authorization": "Bearer <your-runpod-api-key>"}

# 1) 转写一段录音
r = requests.post(API, headers=HEADERS, json={
    "input": {
        "task": "asr",
        "url": "https://dropbox.com/.../recording.m4a",
        "language": "zh",
    }
})
print(r.json()["output"]["text"])

# 2) OCR 一份中文 PDF
r = requests.post(API, headers=HEADERS, json={
    "input": {
        "task": "ocr",
        "url": "https://dropbox.com/.../document.pdf",
    }
})
open("doc.md", "w").write(r.json()["output"]["markdown"])

# 3) MinerU 转一份英文论文
r = requests.post(API, headers=HEADERS, json={
    "input": {
        "task": "mineru",
        "url": "https://arxiv.org/pdf/2310.06825.pdf",
        "backend": "vlm-transformers",  # 用 vlm 模型，最强
    }
})
```

---

## 成本预估 (A40 48GB Flex)

- 单价：$0.00019/s ≈ $0.68/hr
- 一次任务平均 5 分钟（含 1 分钟冷启动 + 4 分钟跑）= $0.057/次
- 月 30 次 = **~$1.7/月**
- 月 100 次 = **~$5.7/月**

Network Volume 存储：~30 GB × $0.07/GB·月 = **$2.1/月**

**月总成本：~$4–10**，完全替代你现在的"手动起 Pod 抢卡 + 装软件"流程。
