import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
import os
import re
from tqdm import tqdm
import json


def generate_subtitle(
    audio_file,
    output_srt=None,
    model_size="medium",
    language="zh",
    silence_threshold=0.05,
    min_silence_duration=0.3,
    precision_mode="high",
):
    """
    将音频文件转换为高精度的 SRT 字幕文件

    参数:
        audio_file (str): 输入音频文件路径
        output_srt (str, optional): 输出 SRT 文件路径
        model_size (str, optional): Whisper 模型大小 ("tiny", "base", "small", "medium", "large")
        language (str, optional): 音频语言代码，默认为中文 "zh"
        silence_threshold (float, optional): 静音检测阈值 (0-1)
        min_silence_duration (float, optional): 最小静音持续时间（秒）
        precision_mode (str, optional): 精度模式 ("standard", "high", "maximum")

    返回:
        str: 生成的字幕文件路径
    """
    # 设置默认输出文件名
    if output_srt is None:
        base_name = os.path.splitext(audio_file)[0]
        output_srt = f"{base_name}.srt"

    # print(f"处理音频文件: {audio_file}")

    # 加载模型和处理器
    model_id = f"openai/whisper-{model_size}"
    processor = WhisperProcessor.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"使用设备: {device}")

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    # 音频加载与分析
    # print("加载和分析音频...")
    input_audio, sampling_rate = librosa.load(audio_file, sr=16000)

    # 计算音频时长
    audio_duration = len(input_audio) / sampling_rate
    # print(f"音频时长: {audio_duration:.2f} 秒")

    # 开始转录过程
    # print("进行音频转录...")
    input_features = processor.feature_extractor(
        input_audio, sampling_rate=16000, return_tensors="pt"
    ).input_features
    input_features = input_features.to(device=device, dtype=torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_features, language=language, task="transcribe"
        )

    # 解码转录文本
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(f"转录文本: {transcription}")

    # 基于精度模式选择分段策略
    segments = []
    if precision_mode in ["high", "maximum"]:
        # 高精度模式: 结合静音检测和语音分析
        segments = enhance_segmentation(
            input_audio,
            sampling_rate,
            transcription,
            language,
            silence_threshold,
            min_silence_duration,
        )
    else:
        # 标准模式: 基于句子分割
        sentences = split_into_sentences(transcription, language)
        # print(f"文本被分割成 {len(sentences)} 个句子")

        if sentences:
            # 分配时间戳
            segments = distribute_timing(sentences, audio_duration)

    # 如无分段，使用完整转录
    if not segments and transcription.strip():
        segments = [{"start": 0, "end": audio_duration, "text": transcription.strip()}]

    # print(f"生成了 {len(segments)} 个字幕分段")

    # 写入 SRT 文件
    with open(output_srt, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()

            # 写入 SRT 格式
            srt_file.write(f"{i+1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

    # 验证文件大小
    file_size = os.path.getsize(output_srt)
    # print(f"字幕文件已保存至: {output_srt} (文件大小: {file_size} 字节)")

    return output_srt


def enhance_segmentation(
    audio, sr, transcription, language, silence_threshold=0.05, min_silence_duration=0.3
):
    """使用静音检测和语音分析增强分段"""
    # print("执行增强分段分析...")

    # 步骤 1: 检测静音段
    # print("检测静音段...")
    # 计算音频振幅包络线
    amplitude_envelope = np.abs(audio)

    # 标准化并应用阈值
    amplitude_envelope = amplitude_envelope / np.max(amplitude_envelope)
    silence_mask = amplitude_envelope < silence_threshold

    # 找出连续的静音区域
    silence_regions = []
    silence_start = None
    for i in range(len(silence_mask)):
        if silence_mask[i] and silence_start is None:
            silence_start = i
        elif not silence_mask[i] and silence_start is not None:
            duration = (i - silence_start) / sr
            if duration >= min_silence_duration:
                silence_regions.append({"start": silence_start / sr, "end": i / sr})
            silence_start = None

    # 处理末尾的静音
    if silence_start is not None:
        duration = (len(silence_mask) - silence_start) / sr
        if duration >= min_silence_duration:
            silence_regions.append(
                {"start": silence_start / sr, "end": len(silence_mask) / sr}
            )

    # print(f"检测到 {len(silence_regions)} 个静音区域")

    # 步骤 2: 基于静音区域分割语音段
    speech_regions = []
    if silence_regions:
        last_end = 0
        for region in silence_regions:
            # 如果静音前有语音
            if region["start"] > last_end:
                speech_regions.append({"start": last_end, "end": region["start"]})
            last_end = region["end"]

        # 添加最后一个语音段
        if last_end < len(audio) / sr:
            speech_regions.append({"start": last_end, "end": len(audio) / sr})
    else:
        # 如果没有检测到静音，将整个音频作为一个语音段
        speech_regions.append({"start": 0, "end": len(audio) / sr})

    # print(f"划分出 {len(speech_regions)} 个语音区域")

    # 步骤 3: 将文本与语音段对应
    segments = []

    # 分割成句子
    sentences = split_into_sentences(transcription, language)

    if not sentences:
        # 如果没有句子，直接返回语音段
        for region in speech_regions:
            segments.append(
                {"start": region["start"], "end": region["end"], "text": transcription}
            )
    elif len(sentences) <= len(speech_regions):
        # 如果句子数量小于或等于语音段数量，可以一一对应
        for i, sentence in enumerate(sentences):
            if i < len(speech_regions):
                segments.append(
                    {
                        "start": speech_regions[i]["start"],
                        "end": speech_regions[i]["end"],
                        "text": sentence.strip(),
                    }
                )
    else:
        # 如果句子数量多于语音段，需要合并一些句子
        # 使用贪婪算法分配句子到语音段
        current_region = 0
        current_text = ""

        for sentence in sentences:
            current_text += sentence

            # 如果当前文本大致匹配当前语音段的长度，或者是最后一个语音段
            if (
                len(current_text) / len(transcription)
                > (
                    speech_regions[current_region]["end"]
                    - speech_regions[current_region]["start"]
                )
                / (len(audio) / sr)
            ) or current_region == len(speech_regions) - 1:

                segments.append(
                    {
                        "start": speech_regions[current_region]["start"],
                        "end": speech_regions[current_region]["end"],
                        "text": current_text.strip(),
                    }
                )

                current_text = ""
                current_region += 1

                if current_region >= len(speech_regions):
                    break

        # 处理剩余句子（如果有）
        if current_text and current_region < len(speech_regions):
            segments.append(
                {
                    "start": speech_regions[current_region]["start"],
                    "end": speech_regions[current_region]["end"],
                    "text": current_text.strip(),
                }
            )

    return segments


def split_into_sentences(text, language="zh"):
    """根据语言特定的标点符号将文本分割成句子"""
    if language in ["zh", "ja", "ko"]:
        # 中文、日文、韩文
        pattern = r"[。！？!?.;；,，]+"
    else:
        # 拉丁语系
        pattern = r"[.!?]+"

    # 分割并清理
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def distribute_timing(sentences, audio_duration):
    """根据句子长度比例分配时间戳"""
    segments = []
    total_chars = sum(len(s) for s in sentences)

    if total_chars > 0:
        current_time = 0
        for sentence in sentences:
            if sentence.strip():
                # 根据字符数估算时长
                sentence_duration = (len(sentence) / total_chars) * audio_duration

                # 添加略微重叠以避免不自然的停顿
                segments.append(
                    {
                        "start": max(0, current_time - 0.1),
                        "end": current_time + sentence_duration,
                        "text": sentence.strip(),
                    }
                )
                current_time += sentence_duration

    return segments


def format_timestamp(seconds):
    """将秒数转换为 SRT 格式的时间戳：00:00:00,000"""
    milliseconds = int((seconds % 1) * 1000)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def create_tts(book_id: str, base_path: str):
    # 从环境变量获取线程数
    try:
        num_threads = int(os.getenv("VIDEO_THREADS", "1"))
    except ValueError:
        num_threads = 1  # 默认使用1个线程

    # 获取 data/book/{book_id}/storyboard 目录下的所有json
    storyboard_dir = f"data/book/{book_id}/storyboard"
    if not os.path.exists(storyboard_dir):
        # print(f"小说信息不存在{storyboard_dir}")
        return
    try:
        chapter_files = os.listdir(storyboard_dir)
        chapter_files.sort(key=lambda x: int(x.split(".")[0]))
        chapter_file_paths = [os.path.join(storyboard_dir, f) for f in chapter_files]
    except Exception as e:
        # print(f"读取章节文件失败：{str(e)}")
        return
    # 计算总进度
    total_items = 0
    try:
        for chapter_file_path in chapter_file_paths:
            with open(chapter_file_path, "r", encoding="utf-8") as f:
                chapter_data = json.load(f)
                total_items += len(chapter_data)
    except Exception as e:
        print(f"计算总进度失败：{str(e)}")
        return
    # 创建总进度条
    with tqdm(total=total_items, desc="总进度", unit="图") as pbar:
        for chapter_file_path in chapter_file_paths:
            with open(chapter_file_path, "r", encoding="utf-8") as f:
                chapter_data = json.load(f)
                for item in chapter_data:
                    audio_path = item["audio_path"]
                    # 判断 audio_path 中是否有 data/ 文字 如果有则不处理 没有的话 需要再前面添加上 /data/book/{book_id}/
                    if "data/" in audio_path:
                        continue
                    else:
                        audio_path = f"data/book/{book_id}/{audio_path}"
                    audio_path = os.path.join(base_path, audio_path)
                    generate_subtitle(audio_path, precision_mode="high")
                    pbar.update(1)


if __name__ == "__main__":
    create_tts("1043294775", os.getcwd())
