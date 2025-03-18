import PIL.Image
import numpy as np
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip
import librosa
import soundfile as sf
import os
import tempfile
import time
import logging
import threading
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import json
import concurrent.futures
import random
from PIL import Image

# 设置日志 - 仅记录错误
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv(override=True)

# 线程锁字典，用于防止同时写入同一个JSON文件
json_locks = {}

# Add compatibility patch for ANTIALIAS
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS


def create_video_with_moving_image(
    image_path,
    audio_path,
    output_path,
    move_direction="left",
    portrait_mode=False,
    video_width=None,
    video_height=None,
    move_distance=0.1,
    move_speed=1.0,
    entrance_effect=False,
    entrance_duration=1.0,
    entrance_direction="left",
    audio_speed=1.0,  # 控制音频播放速度
):
    """
    创建一个将图片与音频合成的视频，图片会在音频播放期间以指定速度朝指定方向移动，可选添加入场效果及调整音频速度。
    如果存在同名的.srt字幕文件，会将字幕添加到视频中。

    参数:
        image_path (str): 图片文件路径
        audio_path (str): 音频文件路径
        output_path (str): 输出视频保存路径
        move_direction (str): 图片移动方向，可选值为 "up", "down", "left", "right"，默认为 "left"
        portrait_mode (bool): 是否使用竖屏模式 (750x1280)，默认为横屏模式 (2560x1440)
        video_width (int): 自定义输出视频宽度（优先级高于预设格式）
        video_height (int): 自定义输出视频高度（优先级高于预设格式）
        move_distance (float): 移动距离，以图片尺寸的比例表示，默认为0.1
        move_speed (float): 移动速度，值越大移动越快，默认为1.0
        entrance_effect (bool): 是否启用入场效果，默认为False
        entrance_duration (float): 入场效果持续时间（秒），默认为1.0秒
        entrance_direction (str): 入场方向，可选值为 "up", "down", "left", "right"，默认为 "left"
        audio_speed (float): 音频播放速度，默认为1.0（正常速度），推荐范围0.5-2.0

    返回:
        bool: 处理成功返回True，失败返回False
    """
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from moviepy.editor import (
            AudioFileClip,
            ImageClip,
            CompositeVideoClip,
            TextClip,
        )
        import os
        import tempfile
        import librosa
        import soundfile as sf
        import platform

        # 确定最终尺寸
        if video_width is not None and video_height is not None:
            # 使用自定义尺寸
            final_width = video_width
            final_height = video_height
        else:
            # 使用预设格式
            if portrait_mode:
                # 竖屏模式 (750x1280)
                final_width, final_height = 750, 1280
            else:
                # 横屏模式 (2560x1440)
                final_width, final_height = 2560, 1440

        # 加载原始音频
        original_audio = AudioFileClip(audio_path)

        # 调整音频速度（保持音调不变）
        if audio_speed != 1.0:
            try:
                # 使用librosa进行高质量的音频处理
                # 创建临时文件用于处理
                temp_folder = tempfile.gettempdir()
                temp_input = os.path.join(temp_folder, "temp_input.wav")
                temp_output = os.path.join(temp_folder, "temp_output.wav")

                # 保存原始音频为临时文件
                original_audio.write_audiofile(temp_input, verbose=False, logger=None)

                # 使用librosa加载音频
                y, sr = librosa.load(temp_input, sr=None)

                # 使用librosa的time_stretch函数调整速度（不改变音调）
                y_stretched = librosa.effects.time_stretch(y, rate=audio_speed)

                # 保存处理后的音频
                sf.write(temp_output, y_stretched, sr)

                # 加载处理后的音频
                audio = AudioFileClip(temp_output)

                # 清理临时文件
                try:
                    os.remove(temp_input)
                    os.remove(temp_output)
                except:
                    pass
            except Exception as e:
                print(f"音频速度调整失败，将使用原始音频。错误: {e}")
                audio = original_audio
        else:
            audio = original_audio

        # 获取调整后的音频时长
        audio_duration = audio.duration

        # 加载图片
        image_clip = ImageClip(image_path)

        # 计算基础缩放因子以保持图片比例适配视频尺寸
        width_ratio = final_width / image_clip.size[0]
        height_ratio = final_height / image_clip.size[1]
        base_scale_factor = min(width_ratio, height_ratio)

        # 根据移动速度和移动距离计算所需的额外缩放
        extra_scale = 1.0 + (move_speed * move_distance)

        # 最终缩放因子，基础值为1.4倍
        scale_factor = base_scale_factor * 1.4 * extra_scale

        # 调整图片大小
        resized_image = image_clip.resize(scale_factor)

        # 计算图片与视频框架之间的可用移动距离
        extra_width = resized_image.size[0] - final_width
        extra_height = resized_image.size[1] - final_height

        # 确保额外距离为正值
        extra_width = max(0, extra_width)
        extra_height = max(0, extra_height)

        # 计算安全移动距离（不会导致黑边）
        safe_x_distance = min(
            resized_image.size[0] * move_distance * move_speed, extra_width
        )
        safe_y_distance = min(
            resized_image.size[1] * move_distance * move_speed, extra_height
        )

        # 计算入场效果的偏移量（用于初始位置计算）
        entrance_x_offset = 0
        entrance_y_offset = 0

        if entrance_effect:
            # 根据入场方向计算初始偏移量
            if entrance_direction == "left":
                entrance_x_offset = -resized_image.size[0]  # 从左侧外部进入
                entrance_y_offset = 0
            elif entrance_direction == "right":
                entrance_x_offset = final_width  # 从右侧外部进入
                entrance_y_offset = 0
            elif entrance_direction == "up":
                entrance_x_offset = 0
                entrance_y_offset = -resized_image.size[1]  # 从上方外部进入
            elif entrance_direction == "down":
                entrance_x_offset = 0
                entrance_y_offset = final_height  # 从下方外部进入

        # 创建移动动画函数
        def move_position(t):
            # 处理入场效果
            if entrance_effect and t < entrance_duration:
                # 计算入场进度 (0到1之间)
                entrance_progress = t / entrance_duration

                # 计算正常移动的起始位置（不考虑入场效果）
                normal_start_x = 0
                normal_start_y = 0

                if move_direction == "right":
                    normal_start_x = -safe_x_distance
                elif move_direction == "down":
                    normal_start_y = -safe_y_distance

                # 根据入场方向和进度计算当前位置
                if entrance_direction == "left":
                    # 从左侧进入到正常起始位置
                    return (
                        entrance_x_offset * (1 - entrance_progress)
                        + normal_start_x * entrance_progress,
                        normal_start_y,
                    )
                elif entrance_direction == "right":
                    # 从右侧进入到正常起始位置
                    return (
                        entrance_x_offset * (1 - entrance_progress)
                        + normal_start_x * entrance_progress,
                        normal_start_y,
                    )
                elif entrance_direction == "up":
                    # 从上方进入到正常起始位置
                    return (
                        normal_start_x,
                        entrance_y_offset * (1 - entrance_progress)
                        + normal_start_y * entrance_progress,
                    )
                elif entrance_direction == "down":
                    # 从下方进入到正常起始位置
                    return (
                        normal_start_x,
                        entrance_y_offset * (1 - entrance_progress)
                        + normal_start_y * entrance_progress,
                    )

            # 入场效果后的正常移动
            # 调整进度计算，考虑入场效果后的剩余时间
            if entrance_effect:
                remaining_time = audio_duration - entrance_duration
                if remaining_time <= 0:
                    # 如果入场时间等于或大于音频时长，则固定在起始位置
                    adjusted_progress = 0
                else:
                    # 计算入场后的移动进度
                    t_after_entrance = max(0, t - entrance_duration)
                    adjusted_progress = min(
                        1.0, (t_after_entrance / remaining_time) * move_speed
                    )
            else:
                # 没有入场效果时的原始计算
                adjusted_progress = min(1.0, (t / audio_duration) * move_speed)

            # 根据移动方向返回相应的位置
            if move_direction == "left":
                return (-safe_x_distance * adjusted_progress, 0)
            elif move_direction == "right":
                return (-safe_x_distance * (1 - adjusted_progress), 0)
            elif move_direction == "up":
                return (0, -safe_y_distance * adjusted_progress)
            elif move_direction == "down":
                return (0, -safe_y_distance * (1 - adjusted_progress))
            else:
                # 默认为左移动
                return (-safe_x_distance * adjusted_progress, 0)

        # 应用移动效果到图片
        moving_image = resized_image.set_position(move_position).set_duration(
            audio_duration
        )

        # 合成视频和音频
        final_clip = CompositeVideoClip(
            [moving_image], size=(final_width, final_height)
        )
        final_clip = final_clip.set_audio(audio)

        # 检查同名SRT字幕文件是否存在 - 使用audio_path而不是output_path
        srt_path = os.path.splitext(audio_path)[0] + ".srt"
        if os.path.exists(srt_path):
            try:
                # 解析SRT字幕文件
                subtitles = parse_srt_file(srt_path)

                if subtitles:
                    # 创建字幕剪辑列表
                    subtitle_clips = []

                    # 计算字幕最大宽度（视频宽度的80%）
                    max_subtitle_width = int(final_width * 0.8)

                    # 查找中文字体
                    font = find_chinese_font()
                    if font is None:
                        print("警告：无法找到支持中文的字体，将使用默认字体")
                        font = ImageFont.load_default()

                    # 使用PIL手动生成静态字幕图像
                    for sub in subtitles:
                        try:
                            # 获取字幕时间和文本
                            start_time = sub["start"]
                            end_time = sub["end"]
                            text = sub["text"]
                            duration = end_time - start_time

                            # 创建一个空白图像作为字幕背景
                            img = Image.new(
                                "RGBA", (final_width, final_height), (0, 0, 0, 0)
                            )
                            draw = ImageDraw.Draw(img)

                            # 计算自适应字体大小
                            font_size = calculate_adaptive_font_size(
                                text, max_subtitle_width, font
                            )

                            # 处理文本换行
                            lines = wrap_text(text, max_subtitle_width, font, font_size)

                            # 计算文本总高度
                            line_height = font_size * 1.5
                            text_height = len(lines) * line_height

                            # 计算起始y坐标（底部上方10%位置）
                            y_position = (
                                final_height - text_height - int(final_height * 0.1)
                            )

                            # 绘制字幕
                            for line_idx, line in enumerate(lines):
                                # 计算当前行的x坐标（居中）
                                text_width = draw.textlength(
                                    line, font=font.font_variant(size=font_size)
                                )
                                x_position = (final_width - text_width) // 2
                                y = y_position + line_idx * line_height

                                # 绘制描边
                                for offset_x in [-2, -1, 0, 1, 2]:
                                    for offset_y in [-2, -1, 0, 1, 2]:
                                        if offset_x != 0 or offset_y != 0:
                                            draw.text(
                                                (x_position + offset_x, y + offset_y),
                                                line,
                                                font=font.font_variant(size=font_size),
                                                fill=(0, 0, 0, 255),  # 黑色描边
                                            )

                                # 绘制黄色文本
                                draw.text(
                                    (x_position, y),
                                    line,
                                    font=font.font_variant(size=font_size),
                                    fill=(255, 255, 0, 255),  # 黄色
                                )

                            # 将PIL图像转换为ImageClip
                            img_array = np.array(img)
                            subtitle_img_clip = ImageClip(img_array)

                            # 设置字幕的持续时间和开始时间
                            subtitle_img_clip = subtitle_img_clip.set_start(
                                start_time
                            ).set_duration(duration)

                            # 添加到字幕剪辑列表
                            subtitle_clips.append(subtitle_img_clip)

                        except Exception as e:
                            print(f"处理单个字幕时出错: {e}")
                            import traceback

                            traceback.print_exc()
                            continue

                    # 将所有字幕添加到视频中
                    if subtitle_clips:
                        # 合成所有剪辑
                        all_clips = [final_clip] + subtitle_clips
                        final_clip = CompositeVideoClip(all_clips)

            except Exception as e:
                print(f"添加字幕时出错: {e}")
                import traceback

                traceback.print_exc()

        # 写入输出文件
        final_clip.write_videofile(
            output_path, fps=24, codec="libx264", verbose=False, logger=None
        )

        # 关闭所有剪辑释放资源
        if audio != original_audio:
            audio.close()
        original_audio.close()
        image_clip.close()
        resized_image.close()
        final_clip.close()

        # 处理成功返回True
        return True

    except Exception as e:
        # 捕获所有异常，打印错误信息并返回False
        print(f"视频生成失败，错误: {e}")
        import traceback

        traceback.print_exc()
        return False


# 查找支持中文的字体
def find_chinese_font():
    """
    在系统中查找支持中文的字体

    返回:
        PIL.ImageFont: 找到的字体对象，如果未找到则返回None
    """
    from PIL import ImageFont
    import os
    import platform

    # 更多中文字体名称
    system = platform.system()

    if system == "Windows":
        # Windows系统的字体目录
        font_dirs = ["C:/Windows/Fonts/"]
        chinese_font_names = [
            "msyh.ttc",  # 微软雅黑
            "simhei.ttf",  # 黑体
            "simsun.ttc",  # 宋体
            "simkai.ttf",  # 楷体
            "STKAITI.TTF",  # 华文楷体
            "STFANGSO.TTF",  # 华文仿宋
            "STXIHEI.TTF",  # 华文细黑
            "STZHONGS.TTF",  # 华文中宋
            "msyhbd.ttf",  # 微软雅黑粗体
            "simfang.ttf",  # 仿宋
            "simyou.ttf",  # 幼圆
            "SIMLI.TTF",  # 隶书
            "STLITI.TTF",  # 华文隶书
            "FZSTK.TTF",  # 方正书体
            "FZYTK.TTF",  # 方正姚体
            "STXINWEI.TTF",  # 华文新魏
        ]
    elif system == "Darwin":  # macOS
        # macOS系统的字体目录
        font_dirs = [
            "/System/Library/Fonts/",
            "/Library/Fonts/",
            os.path.expanduser("~/Library/Fonts/"),
        ]
        chinese_font_names = [
            "PingFang.ttc",  # 苹方
            "STHeiti Light.ttc",  # 华文黑体
            "STHeiti Medium.ttc",  # 华文黑体中
            "Hiragino Sans GB.ttc",  # 冬青黑体
            "Hei.ttf",  # 黑体
            "Kai.ttf",  # 楷体
            "AppleGothic.ttf",  # 苹果哥特体
            "AppleMyungjo.ttf",  # 苹果明朝
        ]
    else:  # Linux
        # Linux系统的字体目录
        font_dirs = [
            "/usr/share/fonts/",
            "/usr/local/share/fonts/",
            os.path.expanduser("~/.fonts/"),
        ]
        chinese_font_names = [
            "wqy-microhei.ttc",  # 文泉驿微米黑
            "wqy-zenhei.ttf",  # 文泉驿正黑
            "NotoSansCJK-Regular.ttc",  # 思源黑体
            "NotoSerifCJK-Regular.ttc",  # 思源宋体
            "NotoSansSC-Regular.otf",  # 思源黑体简体中文
            "NotoSerifSC-Regular.otf",  # 思源宋体简体中文
            "droid-fallback.ttf",  # Droid备用字体
            "DroidSansFallbackFull.ttf",  # Droid完整备用字体
        ]

    # 寻找所有可能的字体
    font_size = 24  # 临时字体大小
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            # 优先检查列表中的中文字体
            for font_name in chinese_font_names:
                font_path = os.path.join(font_dir, font_name)
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        return font
                    except Exception:
                        continue

            # 如果列表中的字体都不可用，尝试遍历目录
            try:
                for file in os.listdir(font_dir):
                    if file.endswith((".ttf", ".ttc", ".otf")):
                        font_path = os.path.join(font_dir, file)
                        try:
                            font = ImageFont.truetype(font_path, font_size)
                            return font
                        except Exception:
                            continue
            except Exception:
                continue

    return None


# 计算自适应字体大小
def calculate_adaptive_font_size(text, max_width, font_base, min_size=24, max_size=72):
    """
    计算适应指定宽度的字体大小

    参数:
        text (str): 要渲染的文本
        max_width (int): 最大宽度
        font_base (PIL.ImageFont): 基础字体对象
        min_size (int): 最小字体大小
        max_size (int): 最大字体大小

    返回:
        int: 适合的字体大小
    """
    from PIL import ImageDraw

    # 创建临时图像进行测量
    temp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)

    # 如果文本包含换行符，找出最长的一行
    lines = text.split("\n")
    if not lines:
        return max_size

    max_line = max(lines, key=len)

    # 二分查找最佳字体大小
    low, high = min_size, max_size
    best_size = min_size

    while low <= high:
        mid = (low + high) // 2
        font = font_base.font_variant(size=mid)

        # 测量宽度
        width = temp_draw.textlength(max_line, font=font)

        if width <= max_width * 0.95:  # 留5%的余量
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1

    # 确保字体大小不会太大或太小
    return max(min(best_size, max_size), min_size)


# 文本换行处理
def wrap_text(text, max_width, font_base, font_size):
    """
    根据最大宽度对文本进行换行处理

    参数:
        text (str): 要处理的文本
        max_width (int): 最大宽度
        font_base (PIL.ImageFont): 基础字体对象
        font_size (int): 字体大小

    返回:
        list: 换行后的文本行列表
    """
    from PIL import ImageDraw

    # 使用指定字体大小
    font = font_base.font_variant(size=font_size)

    # 创建临时图像进行测量
    temp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)

    # 如果文本已包含换行符，先按换行符分割
    raw_lines = text.split("\n")
    lines = []

    for raw_line in raw_lines:
        if not raw_line.strip():
            lines.append("")
            continue

        # 测量当前行宽度
        width = temp_draw.textlength(raw_line, font=font)

        # 如果行宽度小于最大宽度，直接添加
        if width <= max_width:
            lines.append(raw_line)
            continue

        # 否则需要拆分行
        # 判断是否是中文（没有空格）
        words = raw_line.split()

        # 如果分割后数量少（可能是中文），按字符拆分
        if len(words) <= 1:
            current_line = ""
            for char in raw_line:
                test_line = current_line + char
                width = temp_draw.textlength(test_line, font=font)

                if width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = char

            if current_line:
                lines.append(current_line)
        else:
            # 英文单词处理
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                width = temp_draw.textlength(test_line, font=font)

                if width <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

    return lines


# 解析SRT字幕文件的函数
def parse_srt_file(srt_path):
    """
    解析SRT字幕文件，返回包含时间和文本的字幕列表

    参数:
        srt_path (str): SRT文件路径

    返回:
        list: 包含字幕信息的字典列表，每个字典包含'start'、'end'和'text'字段
    """
    subtitles = []

    # 辅助函数：将SRT格式的时间转换为秒
    def time_to_seconds(time_str):
        # 格式: 00:00:00,000
        hours, minutes, seconds = time_str.replace(",", ".").split(":")
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    try:
        with open(srt_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # 解析字幕文件
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 跳过空行
            if not line:
                i += 1
                continue

            # 尝试解析字幕编号（数字）
            try:
                int(line)
                # 如果是字幕编号，获取下一行的时间信息
                i += 1
                if i < len(lines):
                    time_line = lines[i].strip()
                    # 解析时间信息（格式: 00:00:00,000 --> 00:00:00,000）
                    if "-->" in time_line:
                        start_time, end_time = time_line.split("-->")
                        start_seconds = time_to_seconds(start_time.strip())
                        end_seconds = time_to_seconds(end_time.strip())

                        # 收集字幕文本
                        text_lines = []
                        i += 1
                        while i < len(lines) and lines[i].strip():
                            text_lines.append(lines[i].strip())
                            i += 1

                        # 将多行文本合并为一行，保留换行符
                        text = "\n".join(text_lines)

                        # 添加到字幕列表
                        subtitles.append(
                            {"start": start_seconds, "end": end_seconds, "text": text}
                        )
                    else:
                        i += 1
                else:
                    i += 1
            except ValueError:
                # 不是字幕编号，跳过
                i += 1

        return subtitles

    except Exception as e:
        print(f"解析SRT文件失败: {e}")
        return []


# 更新JSON文件中的数据
def update_json_with_video_path(chapter_file_path, item_id, video_path):
    # 获取或创建该文件的锁
    if chapter_file_path not in json_locks:
        json_locks[chapter_file_path] = threading.Lock()

    # 使用锁确保线程安全
    with json_locks[chapter_file_path]:
        try:
            # 读取JSON文件
            with open(chapter_file_path, "r", encoding="utf-8") as f:
                chapter_data = json.load(f)

            # 查找对应的项并更新
            for item in chapter_data:
                if item["id"] == item_id:
                    item["video_path"] = video_path
                    break

            # 写回JSON文件
            with open(chapter_file_path, "w", encoding="utf-8") as f:
                json.dump(chapter_data, f, ensure_ascii=False, indent=4)

            return True
        except Exception as e:
            logger.error(f"更新JSON文件失败：{str(e)}")
            return False


# 处理单个条目
def process_item(item, book_id, chapter_file_path, pbar):
    item_id = item["id"]
    text = item["text"]
    # 构建保存路径
    chapter_name = os.path.basename(chapter_file_path).split(".")[0]
    video_dir = f"data/book/{book_id}/video/{chapter_name}"
    video_path = f"data/book/{book_id}/video/{chapter_name}/{item_id}.mp4"
    image_path = f"data/book/{book_id}/images/{chapter_name}/{item_id}.jpg"
    audio_path = f"data/book/{book_id}/audio/{chapter_name}/{item_id}.mp3"
    # 确保目录存在
    os.makedirs(video_dir, exist_ok=True)
    # 检查文件是否已存在
    if os.path.exists(video_path):
        # 检查JSON是否已更新过
        if "video_path" not in item:
            # 文件存在但JSON未更新，更新JSON
            relative_video_path = f"video/{chapter_name}/{item_id}.mp4"
            update_json_with_video_path(chapter_file_path, item_id, relative_video_path)
        pbar.update(1)  # 更新进度条
        return True

    # 生成视频
    video_data = create_video_with_moving_image(
        image_path,
        audio_path,
        video_path,
        move_direction=random.choice(["left", "up", "down", "right"]),
        portrait_mode=os.getenv("PORTRAIT_MODE"),
        video_width=int(os.getenv("VIDEO_WIDTH")) or 750,
        video_height=int(os.getenv("VIDEO_HEIGHT")) or 1280,
        move_distance=float(os.getenv("MOVE_DISTANCE")) or 0.1,
        move_speed=float(os.getenv("MOVE_SPEED")) or 1.0,
        entrance_effect=os.getenv("ENTRANCE_EFFECT"),
        entrance_duration=float(os.getenv("ENTRANCE_DURATION")),
        audio_speed=float(os.getenv("AUDIO_SPEED")) or 1.0,
    )

    # 检查是否生成成功
    if video_data is None:
        logger.error(f"处理项目 {chapter_name}/{item_id} 失败，跳过")
        pbar.update(1)  # 更新进度条
        return False

    try:
        # 更新JSON文件，添加audio_path字段
        relative_video_path = f"/data/book/{book_id}/video/{chapter_name}/{item_id}.mp4"
        update_json_with_video_path(chapter_file_path, item_id, relative_video_path)
    except Exception as e:
        logger.error(f"保存视频文件失败：{str(e)}")
        pbar.update(1)
        return False

    pbar.update(1)  # 更新进度条
    return True


def create_book_video(book_id):
    # 从环境变量获取线程数
    try:
        num_threads = int(os.getenv("VIDEO_THREADS", "1"))
    except ValueError:
        num_threads = 1  # 默认使用1个线程

    # 获取 data/book/{book_id}/storyboard 目录下的所有json
    storyboard_dir = f"data/book/{book_id}/storyboard"
    if not os.path.exists(storyboard_dir):
        logger.error(f"小说信息不存在{storyboard_dir}")
        return
    try:
        chapter_files = os.listdir(storyboard_dir)
        chapter_files.sort(key=lambda x: int(x.split(".")[0]))
        chapter_file_paths = [os.path.join(storyboard_dir, f) for f in chapter_files]
    except Exception as e:
        logger.error(f"读取章节文件失败：{str(e)}")
        return
    # 计算总进度
    total_items = 0
    try:
        for chapter_file_path in chapter_file_paths:
            with open(chapter_file_path, "r", encoding="utf-8") as f:
                chapter_data = json.load(f)
                total_items += len(chapter_data)
    except Exception as e:
        logger.error(f"计算总进度失败：{str(e)}")
        return
    # 创建总进度条
    with tqdm(total=total_items, desc="总进度", unit="图") as pbar:
        # 使用线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 遍历每个章节文件
            for chapter_file_path in chapter_file_paths:
                try:
                    # 读取章节数据
                    with open(chapter_file_path, "r", encoding="utf-8") as f:
                        chapter_data = json.load(f)

                    # 提交任务到线程池
                    futures = []
                    for item in chapter_data:
                        future = executor.submit(
                            process_item, item, book_id, chapter_file_path, pbar
                        )
                        futures.append(future)

                    # 等待所有任务完成
                    concurrent.futures.wait(futures)
                except Exception as e:
                    logger.error(f"处理章节 {chapter_file_path} 失败：{str(e)}")


if __name__ == "__main__":
    create_book_video("1043294775")
