# -*- coding: utf-8 -*-
"""
Fun-CosyVoice3-0.5B-2512 流式 TTS 服务端
========================================
功能: 提供基于 FastAPI 的流式语音合成服务
支持:
  - 流式音频输出 (StreamingResponse)
  - Zero-shot 音色克隆
  - 预置默认音色
  - 并发请求处理

作者: 凌封 aibook.ren(AI全书)
日期: 2025-12
"""
import os
import sys
print(f"DEBUG SYSTARGV: {sys.argv}")
import argparse
import io
import logging
import time
import numpy as np
import wave
from typing import Optional, Generator, Iterable, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading
import torch
import torchaudio

# 设置路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'third_party', 'Matcha-TTS'))

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 多音色配置 (用户可修改此部分)
# ============================================================================
# 格式: {"id": "音色ID", "file": "文件名", "prompt_text": "音频中说的话"}
# 文件放在 deploy/cosyvoice/asset/ 目录下
# 要求音频清晰（5~15秒为佳，不要过长），保存为wav格式，采集率16kHz,单声道，内容任意，但是和prompt_text里的内容完全一致，必须一字不差！
# CosyVoice3 的 prompt_text 必须以 "You are a helpful assistant.<|endofprompt|>" 开头
# ============================================================================
VOICE_CONFIGS = [
    {
        "id": "default",
        "file": "zero_shot_prompt.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    },
    {
        "id": "longyingcheng",
        "file": "longyingcheng_man.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>真不好意思，从小至今，他还从来没有被哪一位异性朋友亲吻过呢。"
    },
    {
        "id": "longyingwan",
        "file": "longyingwan_woman.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>我们将为全球城市的可持续发展贡献力量。"
    },
    {
        "id": "longyingmu",
        "file": "longyingmu_woman.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>您好，我是智能电话助手，很高兴为您服务。请问您需要咨询业务预约办理还是查询信息？"
    },
    {
        "id": "longshu",
        "file": "longshu.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>Technology has made it easier to learn new languages。通过apps和online courses，anyone can start learning中文或者其他语言。"
    },
    {
        "id": "cross_lingual_en",
        "file": "cross_lingual_prompt.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that's coming into the family is a reason why sometimes we don't buy the whole thing."
    }
]
# ============================================================================

# 全局变量
cosyvoice = None
inference_lock = threading.Lock()

# 多音色缓存: {voice_id: {"file": path, "prompt_text": text}}
voice_cache = {}
default_voice_id = "cross_lingual_en"  # 默认使用的音色ID

# 输出采样率 (可通过 --output_sample_rate 配置)
# 默认 16000 以兼容小智平台
output_sample_rate = 16000

# [注意] CosyVoice API 不支持传入 Tensor，必须传路径或文件对象，因此移除 resampler_cache 优化


app = FastAPI(title="CosyVoice TTS Server", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)





class OpenAISpeechRequest(BaseModel):
    model: str = Field(default="cosyvoice-tts")
    input: str = Field(..., description="The text to generate audio for")
    voice: Optional[str] = Field(default=None, description="Voice ID mapped to CosyVoice voice_id")
    response_format: str = Field(default="wav", description="wav or pcm")
    speed: float = Field(default=1.0, description="Reserved for compatibility; currently ignored")
    stream: bool = Field(default=False, description="When true, stream PCM audio chunks")


def pcm_chunks_to_wav_bytes(chunks: Iterable[bytes], sample_rate: int) -> bytes:
    """Wrap PCM16 mono chunks into a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for chunk in chunks:
            wav_file.writeframes(chunk)
    return buffer.getvalue()


def create_openai_audio_response(audio_chunks: Iterable[bytes], response_format: str, sample_rate: int):
    response_format = (response_format or "wav").lower()
    if response_format == "wav":
        audio_bytes = pcm_chunks_to_wav_bytes(audio_chunks, sample_rate)
        media_type = "audio/wav"
        filename = "speech.wav"
    elif response_format == "pcm":
        audio_bytes = b"".join(audio_chunks)
        media_type = "application/octet-stream"
        filename = "speech.pcm"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'. Supported values: wav, pcm"
        )

    return StreamingResponse(
        iter([audio_bytes]),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Sample-Rate": str(sample_rate),
            "X-Channels": "1",
            "X-Bits": "16"
        }
    )


def create_openai_streaming_response(audio_chunks: Iterable[bytes], response_format: str, sample_rate: int):
    response_format = (response_format or "pcm").lower()
    if response_format != "pcm":
        raise HTTPException(
            status_code=400,
            detail="Streaming is only supported with response_format='pcm'"
        )
    return StreamingResponse(
        audio_chunks,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": 'attachment; filename="speech.pcm"',
            "X-Sample-Rate": str(sample_rate),
            "X-Channels": "1",
            "X-Bits": "16",
            "X-Accel-Buffering": "no"
        }
    )


def resolve_voice_request(voice_id: Optional[str]) -> Optional[str]:
    """Map OpenAI-compatible voice input to an existing CosyVoice voice ID."""
    if not voice_id:
        return default_voice_id
    if voice_id in voice_cache:
        return voice_id
    if voice_id == "alloy" and default_voice_id in voice_cache:
        return default_voice_id
    raise HTTPException(
        status_code=400,
        detail=f"Unknown voice '{voice_id}'. Available voices: {sorted(voice_cache.keys())}"
    )


def generate_audio_stream(
    text: str,
    voice_id: str = None,
    prompt_text: str = None,
    prompt_wav = None,
    stream: bool = True
) -> Generator[bytes, None, None]:
    """
    生成流式音频数据
    
    Args:
        text: 要合成的文本
        voice_id: 音色ID (使用预加载的缓存音色，零延迟)
        prompt_text: 自定义音色的提示文本 (与 prompt_wav 配合使用)
        prompt_wav: 自定义音色的参考音频 (与 prompt_text 配合使用)
        stream: 是否流式输出
    """
    global cosyvoice, voice_cache
    
    with inference_lock:
        try:
            # 确定使用哪个音色
            spk_id = ""
            actual_prompt_text = prompt_text
            actual_prompt_wav = prompt_wav
            
            # 优先使用 voice_id (预缓存音色)
            if voice_id and voice_id in voice_cache:
                spk_id = voice_id
                voice_info = voice_cache[voice_id]
                actual_prompt_text = voice_info["prompt_text"]
                actual_prompt_wav = voice_info["file"]
                logger.debug(f"⚡ 使用预缓存音色: {voice_id} (零I/O/计算)")
            elif voice_id and voice_id not in voice_cache:
                logger.warning(f"音色 '{voice_id}' 不存在，使用默认音色")
                if default_voice_id in voice_cache:
                    spk_id = default_voice_id
                    voice_info = voice_cache[default_voice_id]
                    actual_prompt_text = voice_info["prompt_text"]
                    actual_prompt_wav = voice_info["file"]
            elif prompt_text and prompt_wav:
                # 使用自定义音色 (无缓存，需实时计算)
                logger.debug("使用自定义音色 (实时计算特征)")
            else:
                # 使用默认音色
                if default_voice_id in voice_cache:
                    spk_id = default_voice_id
                    voice_info = voice_cache[default_voice_id]
                    actual_prompt_text = voice_info["prompt_text"]
                    actual_prompt_wav = voice_info["file"]
                    logger.debug(f"⚡ 使用默认音色: {default_voice_id}")
                
            for result in cosyvoice.inference_zero_shot(
                text, 
                actual_prompt_text, 
                actual_prompt_wav,
                stream=stream,
                zero_shot_spk_id=spk_id
            ):
                audio_tensor = result['tts_speech']
                
                # [GPU 重采样] 如果输出采样率与模型原生不同，进行重采样
                if output_sample_rate != cosyvoice.sample_rate:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, 
                        orig_freq=cosyvoice.sample_rate, 
                        new_freq=output_sample_rate
                    )
                
                # [GPU 版本] PCM 16bit 转换
                # 1. GPU 进行乘法 (* 32768)
                # 2. GPU 进行类型转换 (float -> int16)
                # 3. 传输 int16 (2 bytes) 到 CPU
                yield (audio_tensor * 32768).to(torch.int16).cpu().numpy().tobytes()
        except Exception as e:
            logger.error(f"TTS 生成失败: {e}")
            raise


@app.get("/health")
async def health_check():
    """健康检查"""
    # import torch # 已在全局导入
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f}GB"
        }
    
    return JSONResponse({
        "status": "ok",
        "model": "Fun-CosyVoice3-0.5B-2512",
        "openai_compatible_model": "cosyvoice-tts",
        "model_sample_rate": cosyvoice.sample_rate if cosyvoice else None,
        "output_sample_rate": output_sample_rate,
        "available_voices": list(voice_cache.keys()),
        "default_voice": default_voice_id,
        **gpu_info
    })


@app.post("/tts/stream")
async def tts_stream(
    text: str = Form(..., description="要合成的文本"),
    voice_id: Optional[str] = Form(default=None, description="音色ID (使用预加载的音色，零延迟)"),
    prompt_text: Optional[str] = Form(default=None, description="自定义音色的提示文本"),
    prompt_wav: Optional[UploadFile] = File(default=None, description="自定义音色的参考音频")
):
    """
    流式 TTS 接口
    
    参数优先级:
    1. voice_id: 使用预加载的音色 (推荐，零延迟)
    2. prompt_text + prompt_wav: 自定义音色 (需实时计算特征)
    3. 都不传: 使用默认音色
    
    返回: 流式 PCM 音频数据 (采样率由 --output_sample_rate 控制, 16bit, Mono)
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    # 处理自定义音色的参考音频
    prompt_wav_data = None
    if prompt_wav is not None:
        prompt_wav_data = prompt_wav.file
    
    # 记录日志
    if voice_id:
        logger.info(f"TTS 请求: text='{text[:50]}...', voice_id='{voice_id}'")
    elif prompt_text:
        logger.info(f"TTS 请求: text='{text[:50]}...', prompt_text='{prompt_text[:30]}...' (自定义音色)")
    else:
        logger.info(f"TTS 请求: text='{text[:50]}...', voice_id='default'")
    
    start_time = time.time()
    
    def stream_generator():
        first_chunk = True
        total_bytes = 0
        for chunk in generate_audio_stream(
            text, 
            voice_id=voice_id,
            prompt_text=prompt_text, 
            prompt_wav=prompt_wav_data, 
            stream=True
        ):
            if first_chunk:
                logger.info(f"⚡ 首帧延迟: {(time.time() - start_time) * 1000:.0f}ms")
                first_chunk = False
            total_bytes += len(chunk)
            yield chunk
        logger.info(f"✅ TTS 完成: 总耗时 {(time.time() - start_time) * 1000:.0f}ms, 数据量 {total_bytes / 1024:.1f}KB")
    
    return StreamingResponse(
        stream_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(output_sample_rate),
            "X-Channels": "1",
            "X-Bits": "16"
        }
    )


@app.get("/v1/models")
async def list_models():
    created_at = int(time.time())
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": "cosyvoice-tts",
                "object": "model",
                "created": created_at,
                "owned_by": "local"
            }
        ]
    })


@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str):
    if model_id != "cosyvoice-tts":
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return JSONResponse({
        "id": "cosyvoice-tts",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local"
    })


@app.post("/v1/audio/speech")
async def openai_audio_speech(request: OpenAISpeechRequest):
    """
    OpenAI-compatible speech API.

    Example body:
    {
        "model": "cosyvoice-tts",
        "input": "你好，我是小智",
        "voice": "default",
        "response_format": "wav"
    }
    """
    if request.model != "cosyvoice-tts":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{request.model}'. Use 'cosyvoice-tts'."
        )
    if not request.input or len(request.input.strip()) == 0:
        raise HTTPException(status_code=400, detail="input cannot be empty")
    if request.speed <= 0:
        raise HTTPException(status_code=400, detail="speed must be greater than 0")

    voice_id = resolve_voice_request(request.voice)
    logger.info(
        "OpenAI speech request: model='%s', voice='%s', format='%s', stream=%s, text='%s...'",
        request.model,
        voice_id,
        request.response_format,
        request.stream,
        request.input[:50]
    )

    start_time = time.time()
    if request.stream:
        def streaming_generator():
            first_chunk = True
            chunk_count = 0
            total_bytes = 0
            for chunk in generate_audio_stream(
                request.input,
                voice_id=voice_id,
                stream=True
            ):
                if first_chunk:
                    logger.info("⚡ OpenAI stream first chunk: %.0fms", (time.time() - start_time) * 1000)
                    first_chunk = False
                chunk_count += 1
                total_bytes += len(chunk)
                yield chunk
            logger.info(
                "✅ OpenAI speech stream complete: total time %.0fms, chunks %d, bytes %.1fKB",
                (time.time() - start_time) * 1000,
                chunk_count,
                total_bytes / 1024
            )

        return create_openai_streaming_response(
            streaming_generator(),
            request.response_format,
            output_sample_rate
        )

    audio_chunks = list(
        generate_audio_stream(
            request.input,
            voice_id=voice_id,
            stream=True
        )
    )
    logger.info(
        "✅ OpenAI speech complete: total time %.0fms, chunks %d",
        (time.time() - start_time) * 1000,
        len(audio_chunks)
    )
    return create_openai_audio_response(audio_chunks, request.response_format, output_sample_rate)


@app.post("/tts/zero_shot")
async def tts_zero_shot(
    text: str = Form(..., description="要合成的文本"),
    prompt_text: str = Form(..., description="参考音频对应的文本"),
    prompt_wav: UploadFile = File(..., description="参考音频文件 (WAV, 16kHz)")
):
    """
    Zero-shot 音色克隆接口
    
    返回: 流式 PCM 音频数据
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="文本不能为空")
    if not prompt_text or len(prompt_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="提示文本不能为空")
    
    try:
        # 直接传入文件对象 (SpooledTemporaryFile)
        prompt_wav_data = prompt_wav.file
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音频文件加载失败: {e}")
    
    logger.info(f"Zero-shot TTS: text='{text[:50]}...', prompt='{prompt_text[:30]}...'")
    
    def stream_generator():
        for chunk in generate_audio_stream(
            text,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav_data,
            stream=True
        ):
            yield chunk
    
    return StreamingResponse(
        stream_generator(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(output_sample_rate),
            "X-Channels": "1",
            "X-Bits": "16"
        }
    )


def load_model(model_dir: str, device: str = "cuda", fp16: bool = False, use_vllm: bool = False):
    """加载 CosyVoice 模型"""
    global cosyvoice, voice_cache
    
    from cosyvoice.cli.cosyvoice import AutoModel
    
    logger.info(f"正在加载模型: {model_dir}")
    logger.info(f"设备: {device}, FP16: {fp16}, vLLM加速: {use_vllm}")
    
    if use_vllm:
        try:
            import vllm
        except ImportError:
            logger.error("启用 vLLM 失败: 未找到 vllm 库。请先安装: pip install vllm==0.9.0")
            sys.exit(1)
            
    start_time = time.time()
    try:
        # load_vllm 参数会自动触发 CosyVoice2/3 模型的 vLLM 加载逻辑
        # 注意: 库内部默认 gpu_memory_utilization=0.2 (适配 24G 显存)
        cosyvoice = AutoModel(model_dir=model_dir, fp16=fp16, load_vllm=use_vllm)
    except TypeError as e:
        if "load_vllm" in str(e):
             logger.error("当前 CosyVoice 版本似乎不支持 vLLM，请确保使用最新代码")
        raise e
        
    logger.info(f"模型加载完成，耗时: {time.time() - start_time:.1f}s")
    logger.info(f"模型采样率: {cosyvoice.sample_rate}Hz, 输出采样率: {output_sample_rate}Hz")
    
    # ========== 加载多音色配置 ==========
    asset_dir = os.path.join(SCRIPT_DIR, "asset")
    official_asset_dir = os.path.join(SCRIPT_DIR, "official", "asset")
    
    logger.info(f"⚡ 正在加载 {len(VOICE_CONFIGS)} 个音色配置...")
    
    for voice_config in VOICE_CONFIGS:
        voice_id = voice_config["id"]
        voice_file = voice_config["file"]
        prompt_text = voice_config["prompt_text"]
        
        # 查找音频文件
        voice_path = os.path.join(asset_dir, voice_file)
        if not os.path.exists(voice_path):
            # 尝试从官方目录查找
            voice_path = os.path.join(official_asset_dir, voice_file)
        
        if not os.path.exists(voice_path):
            logger.warning(f"❌ 音色 '{voice_id}' 的文件未找到: {voice_file}")
            continue
        
        try:
            # 缓存音色特征
            cosyvoice.add_zero_shot_spk(prompt_text, voice_path, voice_id)
            
            # 保存到 voice_cache
            voice_cache[voice_id] = {
                "file": voice_path,
                "prompt_text": prompt_text
            }
            logger.info(f"✅ 音色 '{voice_id}' 加载成功: {voice_file}")
        except Exception as e:
            logger.warning(f"❌ 音色 '{voice_id}' 加载失败: {e}")
    
    logger.info(f"⚡ 音色加载完成，共 {len(voice_cache)} 个可用音色: {list(voice_cache.keys())}")
    
    # 预热推理 (使用第一个可用音色)
    first_voice_path = None
    if voice_cache:
        first_voice_id = list(voice_cache.keys())[0]
        first_voice_path = voice_cache[first_voice_id]["file"]
    warmup_model(first_voice_path, first_voice_id if voice_cache else None)
    
    return cosyvoice


def warmup_model(prompt_wav_path: str = None, voice_id: str = None):
    """预热模型，减少首次请求延迟"""
    global cosyvoice
    
    if cosyvoice is None:
        return
    
    logger.info("🔥 正在预热模型...")
    start_time = time.time()
    
    warmup_text = "预热测试"
    warmup_prompt_text = "预热"
    
    # 如果有参考音频，使用 zero-shot 预热
    if prompt_wav_path and os.path.exists(prompt_wav_path):
        try:
            # 使用指定的 voice_id 进行预热
            spk_id = voice_id if voice_id else "default"
            for _ in cosyvoice.inference_zero_shot(
                warmup_text, 
                warmup_prompt_text, 
                prompt_wav_path,
                stream=False,
                zero_shot_spk_id=spk_id
            ):
                pass
            logger.info(f"✅ 模型预热完成，耗时: {time.time() - start_time:.1f}s")
        except Exception as e:
            logger.warning(f"预热失败 (不影响正常使用): {e}")
    else:
        logger.info("⏭ 跳过预热 (无参考音频)")


def main():
    global output_sample_rate
    
    parser = argparse.ArgumentParser(description="CosyVoice TTS Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=10096, help="监听端口")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="models/Fun-CosyVoice3-0.5B",
        help="模型目录路径"
    )
    parser.add_argument("--device", type=str, default="cuda", help="运行设备: cuda 或 cpu")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 推理 (节省显存)")
    parser.add_argument("--use_vllm", action="store_true", help="[优化] 使用 vLLM 加速推理 (需 pip install vllm)")
    parser.add_argument(
        "--output_sample_rate", 
        type=int, 
        default=16000, 
        choices=[16000, 24000],
        help="输出采样率: 16000 (兼容小智平台) 或 24000 (原生高质量)"
    )
    args = parser.parse_args()
    
    # 设置输出采样率
    output_sample_rate = args.output_sample_rate
    
    # 处理相对路径
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(SCRIPT_DIR, args.model_dir)
    
    # 加载模型
    load_model(args.model_dir, args.device, args.fp16, args.use_vllm)
    
    # 启动服务
    logger.info(f"服务已启动: http://{args.host}:{args.port}")
    logger.info(f"健康检查: http://{args.host}:{args.port}/health")
    logger.info(f"TTS 接口: POST http://{args.host}:{args.port}/tts/stream")
    logger.info(f"📢 输出采样率: {output_sample_rate}Hz (模型原生: 24000Hz)")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
