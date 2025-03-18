import requests
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm
import concurrent.futures
import time
import logging
import threading
import random

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


# 调用API来生成音频
def generate_audio(text: str, max_retries=3):
    url = os.getenv("AUDIO_API_URL")
    api_key = os.getenv("AUDIO_API_KEY")
    model = os.getenv("AUDIO_MODEL")
    keys = api_key.split(",")
    random_key = random.choice(keys)

    payload = {
        "model": model,
        "input": text,
        "voice": "FunAudioLLM/CosyVoice2-0.5B:benjamin",
        "response_format": "mp3",
        "sample_rate": 44100,
    }
    headers = {
        "Authorization": f"Bearer {random_key}",
        "Content-Type": "application/json",
    }

    for retry in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.content
            else:
                time.sleep(1)  # 休息一秒再重试
        except Exception as e:
            if retry == max_retries - 1:  # 只在最后一次重试失败时记录日志
                logger.error(f"生成音频出错：{str(e)}")
            time.sleep(1)

    return None


# 更新JSON文件中的数据
def update_json_with_audio_path(chapter_file_path, item_id, audio_path):
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
                    item["audio_path"] = audio_path
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
    audio_dir = f"data/book/{book_id}/audio/{chapter_name}"
    audio_path = f"{audio_dir}/{item_id}.mp3"

    # 确保目录存在
    os.makedirs(audio_dir, exist_ok=True)

    # 检查文件是否已存在
    if os.path.exists(audio_path):
        # 检查JSON是否已更新过
        if "audio_path" not in item:
            # 文件存在但JSON未更新，更新JSON
            relative_audio_path = f"audio/{chapter_name}/{item_id}.mp3"
            update_json_with_audio_path(chapter_file_path, item_id, relative_audio_path)
        pbar.update(1)  # 更新进度条
        return True

    # 生成音频
    audio_data = generate_audio(text)

    # 检查是否生成成功
    if audio_data is None:
        logger.error(f"处理项目 {chapter_name}/{item_id} 失败，跳过")
        pbar.update(1)  # 更新进度条
        return False

    # 保存音频文件
    try:
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        # 更新JSON文件，添加audio_path字段
        relative_audio_path = f"/data/book/{book_id}/audio/{chapter_name}/{item_id}.mp3"
        update_json_with_audio_path(chapter_file_path, item_id, relative_audio_path)
    except Exception as e:
        logger.error(f"保存音频文件失败：{str(e)}")
        pbar.update(1)
        return False

    pbar.update(1)  # 更新进度条
    return True


def create_book_audio(book_id: str):
    # 从环境变量获取线程数
    try:
        num_threads = int(os.getenv("AUDIO_THREADS", "1"))
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
    create_book_audio("1043294775")
