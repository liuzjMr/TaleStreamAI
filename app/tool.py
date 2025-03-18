import re


def clean_text(text):
    """
    清理文本中的换行符和多余空格
    
    参数:
        text (str): 需要清理的文本
    
    返回:
        str: 清理后的文本
    """
    if not text:
        return ""
    
    # 移除换行符
    text = text.replace('\n', '')
    
    # 移除多余空格（将多个空格替换为单个空格）
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

def extract_chapter_id_and_name(title):
    """
    从章节标题中提取章节编号和名称
    
    参数:
        title (str): 章节完整标题
    
    返回:
        tuple: (章节编号, 章节名称)
    """
    # 使用正则表达式匹配"第X章"格式
    match = re.search(r'^(第\d+章)', title)
    
    if match:
        # 提取章节编号（包括"章"字）
        chapter_id = match.group(1)
        # 章节名称为剩余部分（去除前后空格）
        chapter_name = title[len(chapter_id):].strip()
        return chapter_id, chapter_name
    
    # 如果没有匹配到标准格式，返回空ID和原标题
    return "", title
