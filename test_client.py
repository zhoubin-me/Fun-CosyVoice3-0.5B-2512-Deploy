# -*- coding: utf-8 -*-
"""
CosyVoice TTS 客户端测试脚本
模拟 Java 服务端的流式 HTTP 请求
作者：凌封
来源：https://aibook.ren (AI全书)
"""
import os
import sys
import time
import argparse
import requests

import wave

def test_health(base_url: str):
    """测试健康检查接口"""
    print("\n[1] 健康检查")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            print(f"  ✓ 服务正常: {resp.json()}")
            return True
        else:
            print(f"  ✗ 服务异常: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return False


def test_tts_stream(base_url: str, text: str, output_path: str):
    """测试流式 TTS 接口"""
    print("\n[2] 流式 TTS 测试")
    print(f"  文本: {text}")
    
    start_time = time.time()
    first_chunk_time = None
    total_bytes = 0
    
    # 确保保存为 .wav
    if not output_path.endswith(".wav"):
        output_path += ".wav"
    
    try:
        resp = requests.post(
            f"{base_url}/tts/stream",
            data={"text": text},
            stream=True,
            timeout=60
        )
        
        if resp.status_code != 200:
            print(f"  ✗ 请求失败: {resp.status_code} - {resp.text}")
            return False
        
        # 获取采样率信息
        sample_rate = int(resp.headers.get("X-Sample-Rate", 24000))
        channels = int(resp.headers.get("X-Channels", 1))
        bits = int(resp.headers.get("X-Bits", 16))
        
        print(f"  采样率: {sample_rate}Hz, 通道: {channels},位深: {bits}bit")
        
        # 使用 wave 模块保存 WAV 文件
        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(bits // 8) # 16bit -> 2 bytes
            wav_file.setframerate(sample_rate)
            
            for chunk in resp.iter_content(chunk_size=4800):
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                        print(f"  ⚡ 首帧延迟: {first_chunk_time * 1000:.0f}ms")
                    
                    wav_file.writeframes(chunk)
                    total_bytes += len(chunk)
        
        total_time = time.time() - start_time
        audio_duration = total_bytes / (sample_rate * (bits // 8) * channels)
        
        print(f"  ✓ 接收完成")
        print(f"  ✓ 数据量: {total_bytes / 1024:.1f}KB")
        print(f"  ✓ 音频时长: {audio_duration:.2f}s")
        print(f"  ✓ 总耗时: {total_time:.2f}s")
        print(f"  ✓ WAV 已保存: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 请求异常: {e}")
        return False


def test_openai_speech(base_url: str, text: str, output_path: str, voice: str, stream: bool):
    """测试 OpenAI 兼容的语音接口"""
    print("\n[2] OpenAI Compatible Speech 测试")
    print(f"  文本: {text}")
    print(f"  音色: {voice}")
    print(f"  流式: {'是' if stream else '否'}")

    start_time = time.time()

    if stream:
        if not output_path.endswith(".pcm"):
            output_path += ".pcm"
    else:
        if not output_path.endswith(".wav"):
            output_path += ".wav"

    try:
        resp = requests.post(
            f"{base_url}/v1/audio/speech",
            json={
                "model": "cosyvoice-tts",
                "input": text,
                "voice": voice,
                "response_format": "pcm" if stream else "wav",
                "stream": stream
            },
            stream=stream,
            timeout=60
        )

        if resp.status_code != 200:
            print(f"  ✗ 请求失败: {resp.status_code} - {resp.text}")
            return False

        total_bytes = 0
        first_chunk_time = None
        with open(output_path, "wb") as f:
            if stream:
                for chunk in resp.iter_content(chunk_size=4800):
                    if chunk:
                        if first_chunk_time is None:
                            first_chunk_time = time.time() - start_time
                            print(f"  ⚡ 首帧延迟: {first_chunk_time * 1000:.0f}ms")
                        f.write(chunk)
                        total_bytes += len(chunk)
            else:
                f.write(resp.content)
                total_bytes = len(resp.content)

        total_time = time.time() - start_time
        print(f"  ✓ 接收完成")
        print(f"  ✓ 数据量: {total_bytes / 1024:.1f}KB")
        print(f"  ✓ 总耗时: {total_time:.2f}s")
        print(f"  ✓ 音频已保存: {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ 请求异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="CosyVoice TTS 客户端测试")
    parser.add_argument("--url", type=str, default="http://localhost:10096", help="服务地址")
    parser.add_argument("--text", type=str, default="你好，我是小智，很高兴为您服务。", help="测试文本")
    parser.add_argument("--output", type=str, default="output/client_test.wav", help="输出文件 (.wav)")
    parser.add_argument("--voice", type=str, default="default", help="音色 ID")
    parser.add_argument(
        "--api",
        type=str,
        default="openai",
        choices=["openai", "legacy"],
        help="测试接口类型: openai 或 legacy"
    )
    parser.add_argument("--stream", action="store_true", help="OpenAI 接口使用流式 PCM 输出")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CosyVoice TTS 客户端测试")
    print("=" * 60)
    print(f"服务地址: {args.url}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 测试健康检查
    if not test_health(args.url):
        print("\n服务不可用，请先启动服务: ./start_server.sh")
        return
    
    # 测试 TTS
    if args.api == "openai":
        test_openai_speech(args.url, args.text, args.output, args.voice, args.stream)
    else:
        test_tts_stream(args.url, args.text, args.output)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
