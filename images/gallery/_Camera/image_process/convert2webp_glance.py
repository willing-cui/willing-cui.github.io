#!/usr/bin/env python3
"""
批量图片缩小并转换为WebP格式脚本
功能：
1. 批量处理图片（支持常见格式）
2. 按指定倍数缩小尺寸
3. 转换为WebP格式
4. 可调节压缩质量
5. 保持原始文件结构
"""

import os
import sys
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


def convert_image_to_webp(
        input_path,
        output_path,
        scale_factor=2,
        quality=85,
        preserve_metadata=True
):
    """
    将单张图片转换为WebP格式

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        scale_factor: 缩小倍数
        quality: WebP压缩质量 (1-100)
        preserve_metadata: 是否保留元数据
    """
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 保留图片模式（如RGB, RGBA等）
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGBA')
            elif img.mode == 'L':
                img = img.convert('L')
            else:
                img = img.convert('RGB')

            # 计算新尺寸
            new_size = (
                int(img.width / scale_factor),
                int(img.height / scale_factor)
            )

            # 调整尺寸
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 保存为WebP格式
            save_args = {
                'format': 'WEBP',
                'quality': quality,
                'method': 6,  # 压缩方法 (0-6, 6为最好但最慢)
                'lossless': False
            }

            # 如果保留元数据
            if preserve_metadata and hasattr(img, 'info'):
                exif = img.info.get('exif')
                if exif:
                    save_args['exif'] = exif

            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存图片
            resized_img.save(output_path, **save_args)

            # 获取文件大小信息
            input_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)
            compression_ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0

            return {
                'input': str(input_path),
                'output': str(output_path),
                'original_size': f"{input_size:,} bytes",
                'compressed_size': f"{output_size:,} bytes",
                'compression_ratio': f"{compression_ratio:.1f}%",
                'dimensions': f"{img.width}x{img.height} -> {new_size[0]}x{new_size[1]}",
                'success': True
            }

    except Exception as e:
        return {
            'input': str(input_path),
            'error': str(e),
            'success': False
        }


def process_directory(
        input_dir,
        output_dir,
        scale_factor=2,
        quality=85,
        workers=4,
        preserve_metadata=True,
        skip_existing=False
):
    """
    处理整个目录

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        scale_factor: 缩小倍数
        quality: 压缩质量
        workers: 并行处理线程数
        preserve_metadata: 是否保留元数据
        skip_existing: 是否跳过已存在的文件
    """
    # 支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # 收集所有图片文件
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in supported_formats:
                # 计算输出路径
                rel_path = file_path.relative_to(input_dir)
                output_path = output_dir / rel_path.with_suffix('.webp')

                # 如果跳过已存在文件且输出文件已存在
                if skip_existing and output_path.exists():
                    continue

                image_files.append((file_path, output_path))

    print(f"找到 {len(image_files)} 个图片文件需要处理")
    print(f"输出目录: {output_dir}")
    print(f"缩放倍数: {scale_factor}")
    print(f"压缩质量: {quality}")
    print(f"并行处理线程数: {workers}")
    print(f"保留元数据: {'是' if preserve_metadata else '否'}")
    print(f"跳过已存在: {'是' if skip_existing else '否'}")
    print("-" * 50)

    if not image_files:
        print("没有需要处理的图片文件")
        return []

    # 并行处理图片
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(
                convert_image_to_webp,
                input_path,
                output_path,
                scale_factor,
                quality,
                preserve_metadata
            ): (input_path, output_path)
            for input_path, output_path in image_files
        }

        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_file), 1):
            input_path, output_path = future_to_file[future]
            result = future.result()
            results.append(result)

            if result['success']:
                print(f"[{i}/{len(image_files)}] ✓ {input_path.name}")
                print(f"    尺寸: {result['dimensions']}")
                print(f"    压缩: {result['compressed_size']} ({result['compression_ratio']})")
            else:
                print(f"[{i}/{len(image_files)}] ✗ {input_path.name} - 错误: {result['error']}")

    return results


def process_single_file(
        input_path,
        output_path,
        scale_factor=2,
        quality=85,
        preserve_metadata=True
):
    """
    处理单个文件
    """
    print(f"处理单个文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"缩放倍数: {scale_factor}")
    print(f"压缩质量: {quality}")
    print(f"保留元数据: {'是' if preserve_metadata else '否'}")
    print("-" * 50)

    result = convert_image_to_webp(
        input_path,
        output_path,
        scale_factor,
        quality,
        preserve_metadata
    )

    if result['success']:
        print(f"\n✓ 处理完成:")
        print(f"  输出: {result['output']}")
        print(f"  尺寸: {result['dimensions']}")
        print(f"  压缩: {result['compressed_size']} ({result['compression_ratio']})")
    else:
        print(f"\n✗ 处理失败: {result['error']}")

    return result


def print_summary(results):
    """打印处理结果摘要"""
    print("\n" + "=" * 50)
    print("处理完成摘要")
    print("=" * 50)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"✓ 成功处理: {len(successful)} 个文件")
    print(f"✗ 处理失败: {len(failed)} 个文件")

    if successful:
        # 计算总体压缩率
        total_input_size = 0
        total_output_size = 0

        print("\n成功处理文件详情:")
        for result in successful:
            # 提取文件大小（移除逗号分隔符）
            input_size_str = result['original_size'].replace(',', '').replace(' bytes', '')
            output_size_str = result['compressed_size'].replace(',', '').replace(' bytes', '')

            try:
                total_input_size += int(input_size_str)
                total_output_size += int(output_size_str)
            except ValueError:
                pass

            print(f"  {Path(result['input']).name}:")
            print(f"    原始: {result['original_size']}")
            print(f"    压缩: {result['compressed_size']} ({result['compression_ratio']})")
            print(f"    尺寸: {result['dimensions']}")

        if total_input_size > 0:
            total_ratio = (1 - total_output_size / total_input_size) * 100
            print(f"\n总计压缩效果:")
            print(f"  原始总大小: {total_input_size:,} bytes ({total_input_size / 1024 / 1024:.2f} MB)")
            print(f"  压缩总大小: {total_output_size:,} bytes ({total_output_size / 1024 / 1024:.2f} MB)")
            print(f"  总体压缩率: {total_ratio:.1f}%")

    if failed:
        print("\n失败文件:")
        for result in failed:
            print(f"  {Path(result['input']).name}: {result['error']}")


def main():
    """
    主函数 - 在这里直接修改参数
    """
    # ========== 在这里修改参数 ==========

    # 1. 输入路径（可以是文件或目录）
    input_path = r"D:/GitHub/willing-cui.github.io/images/gallery/_Camera/watermark"  # 修改为你的输入路径

    # 2. 输出路径（可选，如果不指定会自动生成）
    #    如果输入是文件，输出是文件路径
    #    如果输入是目录，输出是目录路径
    output_path = r"D:/GitHub/willing-cui.github.io/images/gallery/glance"  # 修改为你的输出路径，设为 None 则自动生成

    # 3. 图片处理参数
    scale_factor = 8.0  # 缩小倍数
    quality = 50  # WebP压缩质量 (1-100)，85是较好的平衡点

    # 4. 并行处理参数
    workers = 4  # 并行处理线程数，根据CPU核心数调整

    # 5. 其他选项
    preserve_metadata = True  # 是否保留EXIF等元数据
    skip_existing = True  # 是否跳过已存在的文件

    # ========== 参数设置结束 ==========

    print("图片批量处理工具")
    print("=" * 50)
    print("当前参数设置:")
    print(f"  输入路径: {input_path}")
    print(f"  输出路径: {output_path if output_path else '自动生成'}")
    print(f"  缩小倍数: {scale_factor}")
    print(f"  压缩质量: {quality}")
    print(f"  线程数: {workers}")
    print(f"  保留元数据: {preserve_metadata}")
    print(f"  跳过已存在: {skip_existing}")
    print("=" * 50)

    # 验证参数
    if scale_factor <= 1:
        print("错误: 缩小倍数必须大于1")
        sys.exit(1)

    if not 1 <= quality <= 100:
        print("错误: 压缩质量必须在1-100之间")
        sys.exit(1)

    # 检查输入路径
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        print(f"错误: 输入路径 '{input_path}' 不存在")
        sys.exit(1)

    # 确定输出路径
    if output_path:
        output_path_obj = Path(output_path)
    else:
        if input_path_obj.is_file():
            # 单个文件：在相同目录生成同名.webp文件
            output_path_obj = input_path_obj.with_suffix('.webp')
        else:
            # 目录：在输入目录同级创建新目录
            output_path_obj = input_path_obj.parent / f"{input_path_obj.name}_webp"

    # 处理单个文件
    if input_path_obj.is_file():
        result = process_single_file(
            input_path_obj,
            output_path_obj,
            scale_factor,
            quality,
            preserve_metadata
        )
        results = [result] if result else []

    # 处理目录
    else:
        # 确保输出目录存在
        output_path_obj.mkdir(parents=True, exist_ok=True)

        results = process_directory(
            input_path_obj,
            output_path_obj,
            scale_factor,
            quality,
            workers,
            preserve_metadata,
            skip_existing
        )

    # 打印摘要
    if results:
        print_summary(results)

    print("\n处理完成！")


if __name__ == "__main__":
    main()