import os
import json
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from pathlib import Path


def get_image_creation_time(image_path):
    """
    获取图片的拍摄时间
    优先从EXIF信息中获取，如果无法获取则使用文件创建时间
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()

            if exif_data:
                # EXIF标签中拍摄时间的键值
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == 'DateTimeOriginal' and value:
                        # 将EXIF时间格式转换为标准格式
                        try:
                            # EXIF时间格式通常是: "YYYY:MM:DD HH:MM:SS"
                            dt_str = str(value)
                            if ':' in dt_str:
                                dt_obj = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
                                return dt_obj.strftime('%Y/%m/%d')
                        except (ValueError, AttributeError):
                            pass
    except (AttributeError, KeyError, IndexError, OSError):
        pass

    # 如果无法从EXIF获取，使用文件创建时间
    try:
        file_stat = os.stat(image_path)
        create_time = file_stat.st_ctime
        dt_obj = datetime.fromtimestamp(create_time)
        return dt_obj.strftime('%Y/%m/%d')
    except (OSError, AttributeError):
        return "未知时间"


def get_image_files(directory_path):
    """
    获取目录下所有图片文件
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}
    image_files = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files


def organize_images_by_folder(directory_path):
    """
    按文件夹整理图片信息
    """
    photo_data = {}
    time_data = {}

    # 遍历指定目录下的所有文件夹
    for folder_name in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder_name)

        if os.path.isdir(folder_path):
            photo_list = []
            time_list = []

            # 获取文件夹下所有图片文件
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if Path(file).suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
                                                     '.heic'}:
                        file_path = os.path.join(root, file)

                        # 获取文件名
                        photo_list.append(file)

                        # 获取拍摄时间
                        creation_time = get_image_creation_time(file_path)
                        time_list.append(creation_time)

            if photo_list:  # 只添加有图片的文件夹
                photo_data[folder_name] = photo_list
                time_data[folder_name] = time_list

    return photo_data, time_data


def main():
    # 在这里直接修改目录路径
    directory_path = "D:/GitHub/willing-cui.github.io/images/gallery/glance"  # 请修改为你的图片目录路径

    if not os.path.exists(directory_path):
        print(f"错误: 路径 '{directory_path}' 不存在")
        return

    if not os.path.isdir(directory_path):
        print(f"错误: '{directory_path}' 不是一个目录")
        return

    print(f"正在处理目录: {directory_path}")
    print("正在读取图片信息，请稍候...")

    # 整理图片信息
    photo_data, time_data = organize_images_by_folder(directory_path)

    if not photo_data:
        print("未找到任何图片文件")
        return

    # 创建输出数据结构
    output_data = {
        "photo": photo_data,
        "time": time_data
    }

    # 保存为JSON文件
    output_file = "D:/GitHub/willing-cui.github.io/images/gallery/gallery.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"图片信息已保存到: {output_file}")

    # 打印统计信息
    print("\n统计信息:")
    print(f"共找到 {len(photo_data)} 个文件夹:")
    for folder_name, photos in photo_data.items():
        print(f"  {folder_name}: {len(photos)} 张图片")


if __name__ == "__main__":
    main()