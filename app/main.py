import requests
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
import json
import os
import re
from tool import clean_text, extract_chapter_id_and_name
from tqdm import tqdm
import time

load_dotenv()


def get_book_content(book_id: str) -> str:
    """
    获取书籍内容
    """
    try:
        url = f"https://www.qidian.com/book/{book_id}/"

        payload = {}
        headers = {
            "Cookie": os.getenv("COOKIE"),
            "accept": "*/*",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }

        response = requests.request("GET", url, headers=headers, data=payload)

        # 将内容保存至 /data/book/{book_id}/{book_id}.html
        os.makedirs(f"data/book/{book_id}", exist_ok=True)
        with open(f"data/book/{book_id}/{book_id}.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        # 返回书籍文件路径
        return f"data/book/{book_id}/{book_id}.html"
    except Exception as e:
        print(e)
        return False


# 解析书籍章节
def extract_free_chapters(html_file: str, book_id: str) -> list:
    """
    从HTML文件中提取免费章节信息并转为JSON格式

    参数:
        html_file (str): HTML文件路径

    返回:
        list: 包含章节信息的列表，每个章节为一个字典
    """
    try:
        # 读取HTML文件
        with open(html_file, "r", encoding="utf-8") as file:
            html_content = file.read()
            # 解析HTML内容
        soup = BeautifulSoup(html_content, "html.parser")

        # 查找第一个类名为volume-chapters的ul元素
        volume_chapters = soup.find("ul", class_="volume-chapters")
        if not volume_chapters:
            return []

        chapters = []

        # 查找所有章节项
        chapter_items = volume_chapters.find_all("li", class_="chapter-item")
        for item in chapter_items:
            # 获取章节链接
            link = item.find("a", class_="chapter-name")
            if link:
                href = link.get("href")
                title = link.get_text(strip=True)

                # 清理标题中的换行符和多余空格
                title = clean_text(title)

                # 提取章节编号和名称
                chapter_id, chapter_name = extract_chapter_id_and_name(title)

                # 将相对URL转换为绝对URL
                if href and href.startswith("//"):
                    href = "https:" + href

                # 使用要求的字段名格式
                chapter_info = {"id": chapter_id, "name": chapter_name, "url": href}

                chapters.append(chapter_info)

        # 将章节信息转换为JSON格式
        chapters_json = json.dumps(chapters, ensure_ascii=False, indent=4)

        # 将内容保存至 /data/book/{book_id}/{book_id}.json
        os.makedirs(f"data/book/{book_id}", exist_ok=True)
        with open(f"data/book/{book_id}/{book_id}.json", "w", encoding="utf-8") as f:
            f.write(chapters_json)
        return True
    except Exception as e:
        print(e)
        return False


# 获取每一章的详细内容
def get_chapter_content(book_id: str) -> str:
    """
    从HTML文本中提取小说内容并保存到文件

    参数:
        html_text: 包含小说内容的HTML文本
        output_filename: 保存提取内容的文件名
    """
    try:
        # 读取章节信息List
        with open(f"data/book/{book_id}/{book_id}.json", "r", encoding="utf-8") as file:
            chapters = json.load(file)
        # 使用进度条
        for (index, chapter) in tqdm(enumerate(chapters), desc="获取章节内容"):
            if chapter["url"]:
                payload = {}
                headers = {
                    "Cookie": os.getenv("COOKIE"),
                    "accept": "*/*",
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                }
                response = requests.request(
                    "GET", chapter["url"], headers=headers, data=payload
                )
                # 创建BeautifulSoup对象解析HTML
                soup = BeautifulSoup(response.text, "html.parser")
                # 查找main标签 - 小说内容在main标签内
                main_content = soup.find("main", id=lambda x: x and x.startswith("c-"))
                if not main_content:
                    print("未找到小说内容")
                    return
                # 找到所有p标签 - 每个p标签包含一段小说内容
                paragraphs = main_content.find_all("p")
                # 提取每个p标签的文本
                novel_content = [p.get_text() for p in paragraphs]
                # 将内容保存到文件中
                try:
                    # 创建目录
                    os.makedirs(f"data/book/{book_id}/list", exist_ok=True)
                    output_filename = f"data/book/{book_id}/list/{index}.txt"
                    with open(output_filename, "w", encoding="utf-8") as file:
                        for line in novel_content:
                            file.write(line + "\n")
                except Exception as e:
                    print(f"保存文件时出错: {e}")
            time.sleep(0.5)
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    book = get_book_content("1043294775")
    if book:
        extract_free_chapters(book, "1043294775")
        get_chapter_content("1043294775")
    else:
        print("获取书籍内容失败")
