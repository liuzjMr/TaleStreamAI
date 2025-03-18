import requests
import json
import os
import base64
import time
from io import BytesIO
from PIL import Image  # 需要安装 Pillow 库: pip install Pillow
from dotenv import load_dotenv
from tqdm import tqdm  # 用于显示进度条
import gc
import subprocess
import glob


# 加载环境变量
load_dotenv(override=True)

# 用AI生成画面
def create_Image(prompt: str) -> str:
    url = os.getenv("SD_API_URL")

    payload = json.dumps(
        {
            "prompt": f"{prompt}{os.getenv('SD_LORA')}",
            "sampler_name": "Euler",
            "scheduler": "Exponential",
            "sampler_index": "Euler",
            "restore_faces": True,
            "cfg_scale": 7,
            "steps": os.getenv("SD_STEPS"),
            "width": 512,
            "height": 640,
            "batch_size": 1,
            "n_iter": 1,
        }
    )
    headers = {"Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()

    # 检查是否有错误返回
    if "detail" in response_data:
        error_msg = (
            response_data["detail"][0]["msg"] if response_data["detail"] else "未知错误"
        )
        raise Exception(f"生成图片失败: {error_msg}")

    # 检查是否有正确的返回格式
    if "images" in response_data and len(response_data["images"]) > 0:
        image_data = response_data["images"][0]
        del response_data
        gc.collect()
        return image_data
    else:
        raise Exception("生成图片失败: 返回数据格式不正确")


# 调用高清修复
def upscale_image(image_path: str):
    try:
        path = os.path.join(os.getcwd(), image_path)
        # 构建命令
        command = [
            os.path.join(os.getcwd(), "models", "upscayl-bin.exe"),
            "-i",
            path,
            "-o",
            path,
            "-s",
            os.getenv("UPSCAYL_SCALE"),
            "-m",
            os.path.join(os.getcwd(), "models"),
            "-n",
            os.getenv("UPSCAYL_MODEL"),
            "-f",
            os.getenv("UPSCALY_FILE_TYPE"),
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        return False

# 删除log文件
def delete_log_file():
    # 新增功能：删除当前目录下的所有.log文件
    log_files = glob.glob("*.log")
    for log_file in log_files:
        try:
            os.remove(log_file)
        except Exception as e:
            print(f"删除日志文件失败 {log_file}: {e}")
    time.sleep(0.1)

# 保存base64图片到文件（保存为JPG格式）
def save_base64_image(base64_data: str, save_path: str):
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将base64数据解码为图片并保存为JPG
    try:
        # 使用上下文管理器处理BytesIO
        with BytesIO(base64.b64decode(base64_data)) as image_buffer:
            # 使用上下文管理器处理图像
            with Image.open(image_buffer) as image:
                # 保存为JPG
                image = image.convert("RGB")
                image.save(save_path, "JPEG", quality=95)

        # 手动触发垃圾回收
        gc.collect()
        return True
    except Exception as e:
        return str(e)

# 保存错误信息到文件
def save_error_message(error_msg: str, save_path: str):
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存错误信息
    with open(save_path, "w", encoding="utf-8") as error_file:
        error_file.write(error_msg)

# 根据ID获取小说文件和提示词列表
def get_book_content(book_id: str):
    # 读取目录下的所有json文件
    storyboard_dir = f"data/book/{book_id}/storyboard"

    if not os.path.exists(storyboard_dir):
        raise Exception(f"目录不存在: {storyboard_dir}")

    # 按照文件名排序
    chapter_files = os.listdir(storyboard_dir)
    chapter_files.sort(key=lambda x: int(x.split(".")[0]))
    chapter_file_paths = [os.path.join(storyboard_dir, f) for f in chapter_files]

    # 计算总进度
    total_items = 0
    for chapter_file_path in chapter_file_paths:
        with open(chapter_file_path, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)
            total_items += len(chapter_data)

    # 创建总进度条
    with tqdm(total=total_items, desc="总进度", unit="图") as pbar:
        # 处理每个章节文件
        for chapter_file_path in chapter_file_paths:
            # 获取完整的json文件名（不含路径）
            json_filename = os.path.basename(chapter_file_path).split(".")[0]

            # 读取章节数据
            with open(chapter_file_path, "r", encoding="utf-8") as f:
                chapter_data = json.load(f)
                # 根据 id字段从小到大排序
                chapter_data.sort(key=lambda x: int(x["id"]))

            # 标记是否需要更新JSON文件
            json_updated = False

            # 处理章节中的每个对象
            for item_idx, item in enumerate(chapter_data):
                # 获取id，如果不存在则使用索引
                item_id = item.get("id", f"index_{item_idx}")

                # 获取提示词
                if "lensLanguage_end" in item:
                    prompt = item["lensLanguage_end"]
                    # 使用逗号将他们分隔为数组
                    prompt = prompt.split(",")
                    # 从新将他们拼接为字符串,只取前30个 不足30个则全部取
                    prompt = ",".join(prompt[:30])
                    # 修改图片保存目录结构：data/book/{book_id}/images/{json文件名}/{数据中的id}
                    images_base_dir = f"data/book/{book_id}/images"
                    json_file_dir = os.path.join(images_base_dir, json_filename)
                    os.makedirs(json_file_dir, exist_ok=True)

                    # 设置图片保存路径，使用.jpg后缀
                    image_path = os.path.join(json_file_dir, f"{item_id}.jpg")
                    error_path = os.path.join(json_file_dir, f"{item_id}.txt")

                    # 如果图片已存在，跳过
                    if os.path.exists(image_path):
                        # 检查是否需要更新JSON数据中的image_path字段
                        if "image_path" not in item:
                            # 添加相对路径到JSON数据
                            relative_image_path = f"data/book/{book_id}/images/{json_filename}/{item_id}.jpg"
                            item["image_path"] = relative_image_path
                            json_updated = True
                        gc.collect()
                    else:
                        # 尝试生成图片，最多重试3次
                        retry_count = 0
                        success = False
                        error_msg = ""

                        while retry_count < 3 and not success:
                            try:
                                # 生成图片
                                base64_image = create_Image(prompt)

                                # 保存图片
                                save_result = save_base64_image(
                                    base64_image, image_path
                                )
                                del base64_image
                                gc.collect()
                                if save_result is True:
                                    success = True
                                    # 添加image_path字段到JSON数据
                                    relative_image_path = f"data/book/{book_id}/images/{json_filename}/{item_id}.jpg"
                                    item["image_path"] = relative_image_path
                                    json_updated = True
                                    del save_result
                                    gc.collect()
                                else:
                                    error_msg = f"保存图片失败: {save_result}"
                                    retry_count += 1
                                    time.sleep(1)  # 等待1秒后重试
                            except Exception as e:
                                error_msg = (
                                    f"生成图片失败 (尝试 {retry_count+1}/3): {str(e)}"
                                )
                                retry_count += 1
                                time.sleep(2)  # 等待2秒后重试

                        # 如果所有重试都失败，保存错误信息
                        if not success:
                            save_error_message(error_msg, error_path)
                # 更新进度条
                pbar.update(1)
                # 定期释放内存
                if item_idx % 10 == 0:
                    gc.collect()
                # time.sleep(0.1)
            # 如果有更新，保存更新后的JSON文件
            if json_updated:
                with open(chapter_file_path, "w", encoding="utf-8") as f:
                    json.dump(chapter_data, f, ensure_ascii=False, indent=2)
            # 每处理完一个章节，强制清理内存
            del chapter_data
            gc.collect()


def get_book_images(book_id: str):
    # 获取 data/book/{book_id}/storyboard 目录下的所有json
    storyboard_dir = f"data/book/{book_id}/storyboard"
    chapter_files = os.listdir(storyboard_dir)
    chapter_files.sort(key=lambda x: int(x.split(".")[0]))
    chapter_file_paths = [os.path.join(storyboard_dir, f) for f in chapter_files]

    # 计算总进度
    total_items = 0
    for chapter_file_path in chapter_file_paths:
        with open(chapter_file_path, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)
            total_items += len(chapter_data)

    # 创建总进度条
    with tqdm(total=total_items, desc="总进度", unit="图") as pbar:
        # 遍历每个章节文件
        for chapter_file_path in chapter_file_paths:
            # 读取章节数据
            with open(chapter_file_path, "r", encoding="utf-8") as f:
                chapter_data = json.load(f)
                # 根据 id字段从小到大排序
                chapter_data.sort(key=lambda x: int(x["id"]))
                # 遍历每个对象
                for item in chapter_data:
                    if "image_path" in item:
                        image_path = item["image_path"]

                        # 检查图片大小是否超过2MB
                        if os.path.exists(image_path):
                            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                            if file_size_mb > 2:
                                pbar.update(1)
                                # 删除log文件
                                delete_log_file()
                                continue

                        # 高清修复
                        upscale_result = upscale_image(image_path)
                        if upscale_result is True:
                            # 更新进度条
                            pbar.update(1)
                            # 删除log文件
                            delete_log_file()
                        else:
                            # 重试
                            retry_count = 0
                            while retry_count < 3:
                                upscale_result = upscale_image(image_path)
                                if upscale_result is True:
                                    # 更新进度条
                                    pbar.update(1)
                                    # 删除log文件
                                    delete_log_file()
                                    break
                                retry_count += 1
            




if __name__ == "__main__":
    get_book_content("1043294775")
    get_book_images("1043294775")
