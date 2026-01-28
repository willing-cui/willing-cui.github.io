import os
import re


def count_chinese_characters(text):
    """
    统计中文字符数（包括汉字和中文标点）
    """
    # 匹配中文字符（包括汉字和中文标点）
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')
    chinese_chars = chinese_pattern.findall(text)
    return len(chinese_chars)


def count_english_words(text):
    """
    统计英文单词数
    """
    # 匹配英文单词（字母序列，可包含连字符和撇号）
    english_pattern = re.compile(r'\b[a-zA-Z\'-]+\b')
    english_words = english_pattern.findall(text)
    return len(english_words)


def count_total_characters(text):
    """
    统计总字符数（包括空格、标点等）
    """
    return len(text)


def extract_md_to_txt(folder_path, output_file="output.txt"):
    """
    提取文件夹内所有 .md 文件内容，汇总到 .txt 文件，并统计字数
    """
    # 确保输出文件路径是绝对路径
    output_path = os.path.abspath(output_file)

    # 用于存储找到的 .md 文件
    md_files = []

    # 遍历文件夹及其子文件夹，查找所有 .md 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                full_path = os.path.join(root, file)
                md_files.append(full_path)

    if not md_files:
        print(f"在文件夹 {folder_path} 中未找到任何 .md 文件")
        return

    print(f"找到 {len(md_files)} 个 .md 文件")

    # 初始化统计变量
    total_chinese_chars = 0
    total_english_words = 0
    total_chars = 0
    file_stats = []

    # 写入汇总文件
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for md_file in md_files:
            # 写入文件名分隔符
            outfile.write(f"\n{'=' * 60}\n")
            outfile.write(f"文件: {md_file}\n")
            outfile.write(f"{'=' * 60}\n\n")

            try:
                # 读取并写入 .md 文件内容
                with open(md_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)

                    # 确保每个文件内容后有空行
                    if not content.endswith('\n'):
                        outfile.write('\n')

                    # 统计当前文件的字数
                    chinese_count = count_chinese_characters(content)
                    english_count = count_english_words(content)
                    char_count = count_total_characters(content)

                    # 更新总统计
                    total_chinese_chars += chinese_count
                    total_english_words += english_count
                    total_chars += char_count

                    # 记录文件统计信息
                    file_stats.append({
                        'filename': md_file,
                        'chinese_chars': chinese_count,
                        'english_words': english_count,
                        'total_chars': char_count
                    })

                    print(f"已处理: {os.path.basename(md_file)} - "
                          f"中文字符: {chinese_count}, 英文单词: {english_count}, 总字符: {char_count}")

            except Exception as e:
                error_msg = f"读取文件 {md_file} 时出错: {str(e)}\n"
                outfile.write(error_msg)
                print(error_msg)

        # 写入统计信息到文件末尾
        outfile.write(f"\n{'=' * 60}\n")
        outfile.write("字数统计汇总\n")
        outfile.write(f"{'=' * 60}\n\n")

        outfile.write("按文件统计:\n")
        outfile.write("-" * 40 + "\n")
        for stat in file_stats:
            filename = os.path.basename(stat['filename'])
            outfile.write(f"{filename}:\n")
            outfile.write(f"  中文字符: {stat['chinese_chars']}\n")
            outfile.write(f"  英文单词: {stat['english_words']}\n")
            outfile.write(f"  总字符数: {stat['total_chars']}\n\n")

        outfile.write("总计:\n")
        outfile.write("-" * 20 + "\n")
        outfile.write(f"文件总数: {len(md_files)}\n")
        outfile.write(f"总中文字符数: {total_chinese_chars}\n")
        outfile.write(f"总英文单词数: {total_english_words}\n")
        outfile.write(f"总字符数: {total_chars}\n")

    print(f"\n所有 .md 文件内容已汇总到: {output_path}")
    print(f"\n字数统计汇总:")
    print(f"文件总数: {len(md_files)}")
    print(f"总中文字符数: {total_chinese_chars}")
    print(f"总英文单词数: {total_english_words}")
    print(f"总字符数: {total_chars}")


def main():
    # 设置要遍历的文件夹路径
    folder_path = "../34_Huggingface_Transformers_Lib/Code/transformers-main/docs/source/zh"

    if not folder_path:
        folder_path = os.getcwd()

    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径 '{folder_path}' 不存在")
        return

    # 设置输出文件名
    output_file = "output.txt"

    # 执行提取
    extract_md_to_txt(folder_path, output_file)


if __name__ == "__main__":
    main()