# Fun-CosyVoice3-0.5B-2512 Linux 部署指南

本项目提供 Fun-CosyVoice3-0.5B-2512 语音合成服务的简化部署方案，并已适配 OpenAI Compatible API，方便直接对接标准 SDK 或现有调用链路。

## 功能特性

### ✅ 支持的特性

| 特性 | 说明 |
|------|------|
| **vLLM 加速** | 可选启用，推理速度提升 40%+，首帧延迟显著降低 |
| **流式音频输出** | TTS 边生成边返回 PCM 数据，支持实时播放 |
| **OpenAI Compatible API** | 支持 `POST /v1/audio/speech`，兼容常见 OpenAI TTS 客户端调用方式 |
| **可配置采样率** | 支持 16kHz（兼容小智平台）和 24kHz（原生高质量）两种输出 |
| **多音色预加载** | 支持配置多个音色，启动时预缓存特征，运行时零延迟切换 |
| **Zero-Shot 声音复刻** | 只需 5-15 秒参考音频即可克隆任意音色 |
| **GPU 加速重采样** | 采样率转换在 GPU 上完成，高效节省带宽 |
| **Speaker 特征缓存** | 默认音色特征启动时缓存，推理时零 I/O |

### ❌ 不支持的特性

| 特性 | 原因 |
|------|------|
| **流式文本输入** | 当前使用 HTTP POST，需完整文本后才开始合成，流式输入意义不大，除非LLM推理输出很慢 |
| **单实例高并发** | GPU 推理需串行执行，多请求需排队等待 |
| **多 GPU 自动均衡** | 如需大并发，需手动部署多实例 + 负载均衡 |
| **实时音色切换 API** | 非预加载音色需实时计算特征，首帧延迟较高，可以预加载中加入 |


## 环境要求

- GPU: ~5.2GB 显存 (RTX 3090 实测)
- Python: 3.10+
- CUDA: 12.x
- 系统: Linux (Ubuntu/CentOS)

## 目录结构

```
deploy/cosyvoice/
├── README.md                # 本文档
├── requirements.txt         # Python 依赖列表
├── install.sh               # 环境安装脚本
├── download_model.py        # 模型下载脚本
├── cosyvoice_server.py      # FastAPI 流式服务端
├── start_server.sh          # 启动脚本
├── test_inference.py        # 本地推理测试
├── test_client.py           # 客户端测试脚本
├── official/                # CosyVoice 官方源码 (自动克隆)
├── asset/                   # 预置音色文件
└── models/                  # 模型目录 (自动创建)
    └── Fun-CosyVoice3-0.5B/
```

## 快速部署

### 方式一：自动安装（推荐）

```bash
cd /data/cosyvoice
chmod +x install.sh start_server.sh
./install.sh
```

### 方式二：手动安装

```bash
# 1. 创建 Conda 环境
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice

# 2. 升级 pip
pip install --upgrade pip

# 3. 安装 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装 onnxruntime-gpu (从微软源)
pip install onnxruntime-gpu==1.18.0 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 5. 安装其他依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 6. 克隆官方源码
git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git official
cd official && git submodule update --init --recursive && cd ..

# 7. 创建软链接
ln -sf official/cosyvoice ./cosyvoice
ln -sf official/third_party ./third_party
mkdir -p models asset output

# 8. 下载模型
python download_model.py
```

## 启动服务

```bash
conda activate cosyvoice
./start_server.sh
```

服务监听 `0.0.0.0:10096` 端口。

## 验证服务

```bash
# 健康检查
curl http://localhost:10096/health

# 测试 OpenAI Compatible TTS
python test_client.py --text "你好，我是小智"

# 或使用 curl
curl -X POST -F "text=你好" http://localhost:10096/tts/stream -o test.pcm
ffplay -f s16le -ar 24000 -ac 1 test.pcm

# OpenAI Compatible API
curl http://localhost:10096/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"cosyvoice-tts","input":"你好，我是小智","voice":"default","response_format":"wav"}' \
  --output speech.wav

# OpenAI Compatible Streaming API (PCM)
curl http://localhost:10096/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"cosyvoice-tts","input":"你好，我是小智","response_format":"pcm","stream":true}' \
  --output speech.pcm
```

## API 接口

### 健康检查

```
GET /health
响应: {"status": "ok", "model": "Fun-CosyVoice3-0.5B-2512", ...}
```

### 流式 TTS

```
POST /tts/stream
Content-Type: multipart/form-data

参数:
- text (必需): 要合成的文本
- prompt_text (可选): 提示文本
- prompt_wav (可选): 参考音频文件

响应: PCM 16bit, Mono (采样率由 --output_sample_rate 控制，默认 16kHz)
```

### OpenAI Compatible Speech

```
POST /v1/audio/speech
Content-Type: application/json

请求体:
{
  "model": "cosyvoice-tts",
  "input": "你好，我是小智",
  "voice": "default",
  "response_format": "wav"
}

参数:
- model (必需): 固定为 `cosyvoice-tts`
- input (必需): 要合成的文本
- voice (可选): 对应服务端的 voice_id，不传时使用默认音色
- response_format (可选): `wav` 或 `pcm`
- speed (可选): 兼容字段，当前版本暂未生效
- stream (可选): `true` 时启用分块流式输出，仅支持 `response_format="pcm"`

响应:
- `wav`: 标准 WAV 音频文件
- `pcm`: 原始 PCM 16bit Mono 音频流
- `stream=true`: 通过 HTTP chunked transfer 持续返回 PCM 数据
```

### OpenAI Compatible Models

```
GET /v1/models
GET /v1/models/{model_id}
```

当前可用模型:
- `cosyvoice-tts`

### Zero-shot 克隆

```
POST /tts/zero_shot
Content-Type: multipart/form-data

参数:
- text (必需): 要合成的文本
- prompt_text (必需): 参考音频对应的文本
- prompt_wav (必需): 参考音频 (WAV, 16kHz)
```

### vLLM 加速 (可选)

本项目支持使用 `vllm` 库加速推理 (仅限 Linux + CUDA)。

**启用方式**:
1. 安装时选择安装 vLLM (或运行 `pip install vllm==0.9.0`)。
2. 启动服务时添加 `--use_vllm` 参数。

```bash
# 启动服务
python cosyvoice_server.py --use_vllm

# 本地测试
python test_inference.py --use_vllm
```

> [!NOTE]
> vLLM 模式下，默认 GPU 显存占用率限制为 **0.2** (约 4.8GB)，这是为了给 FunASR 和 LLM 预留显存。如需修改，请修改 `cosyvoice/cli/model.py` 中的 `gpu_memory_utilization` 参数。

### 输出采样率配置

CosyVoice3 模型原生输出 **24kHz** 音频，但小智平台使用 **16kHz**。服务端提供了可配置的采样率转换功能，**在 GPU 上高效重采样**。

**启动参数**:
```bash
# 默认 16kHz (兼容小智平台)
./start_server.sh --use_vllm

# 原生高质量 24kHz (用于其他场景)
./start_server.sh --use_vllm --output_sample_rate 24000
```

> [!TIP]
> GPU 重采样效率极高 (使用 `torchaudio.functional.resample`)，同时减少 33% 网络传输数据量。

## 性能优化亮点

## OpenAI SDK 调用示例

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:10096/v1",
)

audio = client.audio.speech.create(
    model="cosyvoice-tts",
    voice="default",
    input="你好，我是小智，很高兴为您服务。",
    response_format="wav",
)

audio.stream_to_file("speech.wav")
```

流式 PCM 示例:

```python
import requests

resp = requests.post(
    "http://localhost:10096/v1/audio/speech",
    json={
        "model": "cosyvoice-tts",
        "input": "你好，我是小智，很高兴为您服务。",
        "response_format": "pcm",
        "stream": True,
    },
    stream=True,
)

with open("speech.pcm", "wb") as f:
    for chunk in resp.iter_content(chunk_size=4800):
        if chunk:
            f.write(chunk)
```

通过针对性的深度优化，本项目在消费级显卡上实现了极致的推理性能：

1.  **vLLM 加速后端 (可选)**
    - **原理**: 引入 vLLM 替代 PyTorch 进行 LLM 推理，利用 PagedAttention 和高度优化的 CUDA Kernels。
    - **效果**: 推理速度提升 **40%+**，显著降低首帧延迟。
    - **显存**: 自动控制显存占用 (4.8GB)，实现与 FunASR 的完美共存。

2.  **零 I/O 特征缓存 (Zero-Shot Speaker Caching)**
    - **原理**: 启动时预先计算默认音色的声学特征 (Emb/Feat) 并常驻内存。
    - **效果**: 默认音色推理实现 **零 I/O / 零特征提取**，极致响应。

3.  **GPU 加速后处理 (GPU PCM Conversion)**
    - **原理**: 将音频 Float32 -> Int16 的转换和归一化操作移至 GPU 执行。
    - **效果**: 降低服务端 CPU 负载，减少 50% 的 GPU-CPU 数据传输带宽。

## 常见问题 (Q&A)

### Q1: 生成的音频是乱音/噪音

**原因**: `transformers` 库版本不对

**解决**:
```bash
pip install transformers==4.51.3
```

### Q2: 报错 `libcudnn.so.8: cannot open shared object file`

**原因**: cuDNN 8.x 库未正确配置，ONNX Runtime 无法启用 GPU 加速。

**解决**:

- **如果是普通部署** (无 vLLM):
  ```bash
  pip install nvidia-cudnn-cu12==8.9.7.29
  export LD_LIBRARY_PATH=... (见 install.sh)
  ```

- **如果启用了 vLLM** (推荐):
  **请忽略此报错！**
  vLLM 依赖新版 cuDNN (v9)，强制降级会导致 vLLM 无法启动。ONNX Runtime 会自动回退到 CPU 运行辅助模型，对整体速度几无影响。

### Q3: 报错 `ModuleNotFoundError: No module named 'xxx'`

**原因**: 依赖未完整安装

**解决**:
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### Q4: NumPy 版本错误 (`_ARRAY_API not found`)

**原因**: 安装了 NumPy 2.x，与 onnxruntime 不兼容

**解决**:
```bash
pip install "numpy<2"
```

### Q5: 首帧延迟高 (~6秒)

**原因**: 首次推理需要编译 CUDA kernels

**解决**:
1. 服务启动时会自动预热，预热后首帧延迟降至 ~2-3秒
2. 考虑使用 TensorRT 加速进一步降低延迟

### Q6: 警告 `synthesis text xxx too short than prompt text`

**原因**: 合成文本比参考音频文本短

**解决**: 这个警告对 CosyVoice3 影响较小，可忽略。如需避免，使用更长的合成文本。

### Q7: 显存不足 (CUDA OOM)

模型实测需要 ~5.2GB 显存，确保 GPU 有足够空间。

### Q8: 为什么要设置 `default_prompt_text`？可以随便写吗？

**不可以。**

`default_prompt_text` 必须与 `default_prompt_wav` (参考音频) 的内容**完全一致**。
CosyVoice 需要确切知道参考音频里说了什么字，才能准确提取音色和韵律特征。如果内容不匹配（例如音频里说的是“你好”，文本写的是“天气不错”），会导致音色克隆失败或产生幻觉。

默认配置：
- **音频**: `official/asset/zero_shot_prompt.wav`
- **文本**: `"You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"`

如需更换默认音色，请同时替换音频文件和修改代码中的 `default_prompt_text`。
要求音频清晰（5~15秒为佳，不要过长），保存为wav格式，采集率16kHz,单声道，内容任意，但是和prompt_text里的内容完全一致，必须一字不差！

**作者**：凌封 | **来源**：[aibook.ren (AI全书)](https://aibook.ren)
