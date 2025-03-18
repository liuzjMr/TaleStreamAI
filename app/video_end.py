import os
import subprocess


def save_output_video(book_id):
    # 使用os.path.join合并路径
    video_dir = os.path.join(os.getcwd(), "data", "book", str(book_id), "video")
    # 设置最终保存位置
    output_file = os.path.join(
        os.getcwd(), "data", "book", str(book_id), str(book_id) + ".mp4"
    )
    # 递归遍历这个目录下的所有视频
    video_paths = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".mp4"):
                # 使用os.path.join合并完整路径
                video_paths.append(os.path.join(root, file))

    # 对视频路径进行排序
    def sort_key(path):
        # 从路径中提取文件夹编号和文件编号
        parts = path.replace("\\", "/").split("/")
        folder_num = int(parts[-2])  # 倒数第二个部分是文件夹编号
        file_num = int(
            parts[-1].split(".")[0]
        )  # 最后一个部分是文件名，去掉.mp4后转为数字
        return (folder_num, file_num)

    video_paths.sort(key=sort_key)

    # 获取concat_list.txt的完整路径
    concat_list_path = os.path.join(os.getcwd(), "concat_list.txt")

    # 如果文件存在，先删除
    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)

    # 将视频路径写入文件，每行前面加上"file "
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for path in video_paths:
            # 将路径转换为正确的格式：
            # 1. 替换所有反斜杠为正斜杠
            # 2. 在路径两边加上单引号，以处理可能包含空格的路径
            formatted_path = path.replace("\\", "/")
            f.write(f"file '{formatted_path}'\n")
    # 添加内存优化参数
    result = subprocess.call(
        [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "concat_list.txt",
            "-c",
            "copy",
            "-max_muxing_queue_size",
            "9999",  # 增加复用队列大小
            "-threads",
            "1",  # 减少线程数以降低内存
            output_file,
        ]
    )
    if result == 0:
        print(f"视频合并成功，保存位置: {output_file}")
    else:
        print(f"视频合并失败，错误码: {result}")


if __name__ == "__main__":
    save_output_video("1043294775")
