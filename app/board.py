import os
import json
import time
from openai import OpenAI
import re
from tqdm import tqdm
from dotenv import load_dotenv



load_dotenv(override=True)


prompt = """
你是一个资深的剧本编辑
请根据我输入的内容生成分镜，分镜要包含所有小说内容，并且严格按照我输入的格式给我，其中 text为分镜文字内容，screenKeywords_cn 为分镜内容的镜头语言中文描述 screenKeywords_en为 镜头语言的英文描述，一个好的镜头语言可能包含这几类
角色，动作，场景，情绪，风格，镜头角度，灯光与环境
角色   年轻男子、老年女性、英雄、反派   描述角色的年龄、外观或角色类型。
动作   跑步、微笑、哭泣、惊讶地看   明确角色的动作或表情。
场景   森林、城市街道、海滩、厨房   指定故事发生的地点或背景。
情绪   快乐、悲伤、神秘、浪漫   设定场景的氛围或情绪基调。
风格   素描、水彩、卡通、写实、动漫   选择图像的艺术风格。
镜头角度   特写、中景、广角、俯视   指定摄像机的视角或构图。
灯光与环境   阳光、雨天、黄昏、夜景、背光  描述光线条件或环境氛围。
不要对分镜文案进行提炼，一些角色人名，可以根据名字推测是男女还是青年少年，提示词中不要用人名
错误例子 
角色：年轻男子，动作：喝酒、沉思，场景：星空下，情绪：孤独、怀念，镜头角度：中景，灯光与环境：星光、夜晚
正确例子 
年轻男子，喝酒、沉思，星空下，孤独、怀念，中景，星光、夜晚
List a few popular cookie recipes in JSON format.
Use this JSON schema:
Recipe =[
    {
        "id": "1",
        "text": "xxxxxx",
        "lensLanguage_cn": "",
        "lensLanguage_en": ""
    },
    {
        "id":"2",
        "text":"xxxxxxxx",
        "lensLanguage_cn":"",
        "lensLanguage_en":""
    }
]
Return: list[Recipe]
"""


def generate_board_json(chapter_content: str, max_retries=3, retry_delay=2):
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=os.getenv("GEMINI_API_URL"),
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chapter_content},
                ],
            )

            content = response.choices[0].message.content
            content = re.sub(r"```json\n?|\n?```", "", content)

            try:
                result = json.loads(content)
                # 验证结果非空
                if result and isinstance(result, list) and len(result) > 0:
                    return result
                else:
                    print(f"API返回空结果，第{attempt+1}次尝试")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    continue
            except json.JSONDecodeError:
                print(f"JSON解析失败，第{attempt+1}次尝试")
                print(f"原始内容: {content[:100]}...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue

        except Exception as e:
            print(f"API请求错误: {str(e)}，第{attempt+1}次尝试")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue

    print("所有重试尝试都失败，返回空列表")
    return []


def split_content_into_chunks(content, chunk_size=100):
    """
    将内容按行分割成多个块

    Args:
        content (str): 要分割的内容
        chunk_size (int): 每个块的最大行数

    Returns:
        list: 分割后的内容块列表
    """
    lines = content.splitlines()
    chunks = []

    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


def merge_json_results(results_list):
    """
    合并多个JSON结果列表为一个列表

    Args:
        results_list (list): 包含多个JSON结果列表的列表

    Returns:
        list: 合并后的JSON结果列表
    """
    merged_results = []
    id_counter = 1

    for results in results_list:
        for item in results:
            # 更新ID以确保连续性
            item["id"] = str(id_counter)
            merged_results.append(item)
            id_counter += 1

    return merged_results


def generate_board(book_id: str):
    # 确保目标目录存在
    storyboard_dir = f"data/book/{book_id}/storyboard"
    if not os.path.exists(storyboard_dir):
        os.makedirs(storyboard_dir)

    # 获取所有章节文件
    chapter_files = os.listdir(f"data/book/{book_id}/list")
    # 按文件名排序
    chapter_files.sort(key=lambda x: int(x.split(".")[0]))

    # 跟踪处理结果
    failed_chapters = []
    skipped_chapters = []

    for chapter_file in tqdm(chapter_files):
        # 获取章节索引
        index = chapter_file.split(".")[0]
        # 检查目标文件是否已存在且有内容
        target_file = f"{storyboard_dir}/{index}.json"

        # 文件存在性检查
        if os.path.exists(target_file):
            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    existing_content = json.load(f)
                # 验证内容有效（非空列表或字典）
                if existing_content and (
                    isinstance(existing_content, list) and len(existing_content) > 0
                ):
                    print(f"跳过章节 {chapter_file} - 文件已存在且内容有效")
                    skipped_chapters.append(chapter_file)
                    continue  # 跳过处理
            except (json.JSONDecodeError, IOError):
                # 如果文件存在但内容无效或不可读，则重新处理
                print(f"文件 {target_file} 存在但包含无效数据 - 重新处理")

        # 读取章节内容
        with open(
            f"data/book/{book_id}/list/{chapter_file}", "r", encoding="utf-8"
        ) as file:
            chapter_content = file.read()

        # 首先尝试处理完整章节
        board_json = generate_board_json(chapter_content)

        # 如果失败了，检查内容长度并可能进行分块处理
        if not board_json:
            lines = chapter_content.splitlines()
            line_count = len(lines)

            if line_count > 120:
                print(f"章节 {chapter_file} 内容过长 ({line_count} 行)，进行分块处理")
                chunks = split_content_into_chunks(chapter_content, 100)
                chunk_results = []

                print(f"将章节分为 {len(chunks)} 个块进行处理")
                for i, chunk in enumerate(chunks):
                    print(f"处理块 {i+1}/{len(chunks)}")
                    chunk_json = generate_board_json(chunk)

                    if chunk_json:
                        chunk_results.append(chunk_json)
                    else:
                        print(f"警告：无法为章节 {chapter_file} 的块 {i+1} 生成分镜")

                if chunk_results:
                    # 合并所有成功的块结果
                    board_json = merge_json_results(chunk_results)
                    print(
                        f"成功合并 {len(chunk_results)} 个块的结果，共 {len(board_json)} 个分镜项"
                    )
                else:
                    print(f"警告：章节 {chapter_file} 的所有块处理都失败了")
            else:
                print(
                    f"章节 {chapter_file} 虽然处理失败，但内容不超过120行，尝试重新生成"
                )
                # 再次尝试处理完整内容
                board_json = generate_board_json(chapter_content)

        # 处理空结果
        if not board_json:
            failed_chapters.append(chapter_file)
            print(f"警告：无法为章节 {chapter_file} 生成分镜")
            continue

        # 将JSON写入文件
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(board_json, f, ensure_ascii=False, indent=2)

    # 报告处理结果
    if skipped_chapters:
        print(f"跳过了 {len(skipped_chapters)} 个章节（文件已存在且内容有效）")

    if failed_chapters:
        print(f"处理完成，但以下章节失败: {', '.join(failed_chapters)}")
        return False
    else:
        processed_count = len(chapter_files) - len(skipped_chapters)
        print(
            f"所有章节处理成功。处理了 {processed_count} 个章节，跳过了 {len(skipped_chapters)} 个章节"
        )
        return True


if __name__ == "__main__":
    success = generate_board("1043294775")
    if not success:
        print("部分章节处理失败。请检查并重试。")
