import os
from app.main import get_book_content, extract_free_chapters, get_chapter_content
from app.board import generate_board
from app.image import get_book_images, get_book_content as create_book_image
from app.audio import create_book_audio
from app.tts import create_tts
from app.video import create_book_video
from app.video_end import save_output_video


if __name__ == "__main__":
    book_id = "1043294775"
    book = get_book_content(book_id)
    if book:
        extract_free_chapters(book, book_id)
        get_chapter_content(book_id)
    else:
        print("获取书籍内容失败")
    success = generate_board(book_id)
    if success:
        # 生成图片
        create_book_image(book_id)
        # 高清修复
        get_book_images(book_id)
        # 生成音频
        create_book_audio(book_id)
        # 生成字幕文件
        create_tts(book_id, os.getcwd())
        # 视频分段生成
        create_book_video(book_id)
        # 视频总合成
        save_output_video(book_id)
