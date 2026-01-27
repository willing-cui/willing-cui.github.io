import os
import colorsys
from PIL import Image, ImageEnhance, ImageStat, ImageOps, ImageFile
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import numpy as np


class WatermarkProcessor:
    def __init__(self, watermark_path, opacity=0.7, position='auto',
                 scale=0.2, margin=20, auto_color=True,
                 min_contrast_ratio=3.0):
        """
        初始化水印处理器

        参数:
            watermark_path: 水印图片路径
            opacity: 水印透明度 (0-1)
            position: 水印位置 ('auto', 'bottom-right', 'bottom-left', 'top-right', 'top-left')
            scale: 水印相对于原图的比例
            margin: 水印距离边缘的边距(像素)
            auto_color: 是否自动调整水印颜色
            min_contrast_ratio: 最小对比度比例
        """
        self.watermark = Image.open(watermark_path).convert("RGBA")
        self.opacity = opacity
        self.position = position
        self.scale = scale
        self.margin = margin
        self.auto_color = auto_color
        self.min_contrast_ratio = min_contrast_ratio

    def get_exif_orientation(self, image):
        """获取图片的EXIF方向信息"""
        try:
            exif = image.getexif()
            if exif is not None:
                for tag, value in exif.items():
                    if TAGS.get(tag) == 'Orientation':
                        print(f"成功读取方向信息: {value}")
                        return value
        except (AttributeError, KeyError, IndexError):
            pass
        print(f"未能成功读取方向信息，返回默认值: 1")
        return 1  # 默认方向

    def correct_image_orientation(self, image):
        """根据EXIF信息校正图片方向"""
        try:
            orientation = self.get_exif_orientation(image)

            if orientation == 1:
                # 正常方向
                return image.copy()
            elif orientation == 2:
                # 水平翻转
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                # 旋转180度
                return image.transpose(Image.ROTATE_180)
            elif orientation == 4:
                # 垂直翻转
                return image.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                # 水平翻转后旋转270度
                return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
            elif orientation == 6:
                # 旋转270度
                return image.transpose(Image.ROTATE_270)
            elif orientation == 7:
                # 水平翻转后旋转90度
                return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
            elif orientation == 8:
                # 旋转90度
                return image.transpose(Image.ROTATE_90)
            else:
                return image.copy()

        except Exception as e:
            print(f"警告: 校正图片方向时出错: {str(e)}")
            return image.copy()

    def reset_exif_orientation(self, exif_bytes):
        """
        将EXIF数据中的方向标签(0x0112)设置为1（正常方向）

        参数:
            exif_bytes: 原始的EXIF字节数据

        返回:
            修改后的EXIF字节数据
        """
        if not exif_bytes:
            return exif_bytes

        try:
            # 创建内存中的图片对象来处理EXIF
            img = Image.new('RGB', (1, 1))
            img.info['exif'] = exif_bytes

            # 获取EXIF数据
            exif = img.getexif()

            # 检查是否有EXIF数据
            if exif is None:
                return exif_bytes

            # 设置方向标签为1（正常方向）
            # 0x0112 是EXIF方向标签的ID
            exif[0x0112] = 1  # 1 = 正常方向

            # 转换为字节
            return exif.tobytes()

        except Exception as e:
            # 如果处理失败，返回原始数据
            print(f"处理EXIF时出错: {e}")
            return exif_bytes

    def calculate_position(self, base_width, base_height, wm_width, wm_height):
        """计算水印位置"""
        if self.position == 'auto':
            # 自动选择位置：如果图片宽大于高，放右下角；否则放底部居中
            if base_width > base_height * 1.2:  # 宽幅图片
                return (base_width - wm_width - self.margin,
                        base_height - wm_height - self.margin)
            else:  # 竖幅或方形图片
                return ((base_width - wm_width) // 2,
                        base_height - wm_height - self.margin)

        positions = {
            'bottom-right': (base_width - wm_width - self.margin,
                             base_height - wm_height - self.margin),
            'bottom-left': (self.margin,
                            base_height - wm_height - self.margin),
            'top-right': (base_width - wm_width - self.margin, self.margin),
            'top-left': (self.margin, self.margin)
        }

        return positions.get(self.position, positions['bottom-right'])

    def analyze_background_color(self, image, position, wm_width, wm_height):
        """分析水印放置位置的背景颜色"""
        x, y = position

        # 获取水印区域的边界
        left = max(0, x - 5)  # 稍微扩大区域以获得更好的分析
        upper = max(0, y - 5)
        right = min(image.width, x + wm_width + 5)
        lower = min(image.height, y + wm_height + 5)

        # 截取背景区域
        bg_region = image.crop((left, upper, right, lower))

        if bg_region.mode != 'RGB':
            bg_region = bg_region.convert('RGB')

        # 转换为numpy数组进行分析
        bg_array = np.array(bg_region)

        # 计算区域的平均颜色
        avg_color = np.mean(bg_array, axis=(0, 1))

        # 计算区域的亮度
        brightness = np.mean(avg_color) / 255.0

        # 计算颜色的标准差，判断背景复杂度
        std_color = np.std(bg_array, axis=(0, 1)).mean() / 255.0

        return {
            'color': tuple(avg_color.astype(int)),
            'brightness': brightness,
            'complexity': std_color,
            'is_dark': brightness < 0.5
        }

    def get_contrast_color(self, bg_color_info):
        """根据背景颜色计算对比度最高的水印颜色"""
        bg_color = bg_color_info['color']
        bg_brightness = bg_color_info['brightness']

        # 将RGB转换为HSL
        r, g, b = [c / 255.0 for c in bg_color]
        h, l, s = colorsys.rgb_to_hls(r, g, b)

        # 根据背景亮度选择颜色
        if bg_brightness < 0.4:  # 暗色背景
            # 使用亮色水印
            wm_color = (255, 255, 255)  # 白色
        elif bg_brightness > 0.6:  # 亮色背景
            # 使用暗色水印
            wm_color = (0, 0, 0)  # 黑色
        else:  # 中等亮度背景
            # 计算对比度更高的颜色
            if l < 0.5:
                wm_color = (255, 255, 255)  # 白色
            else:
                wm_color = (0, 0, 0)  # 黑色

            # 检查对比度是否足够
            contrast = self.calculate_contrast_ratio(bg_color, wm_color)
            if contrast < self.min_contrast_ratio:
                # 对比度不足，尝试使用互补色
                complementary_h = (h + 0.5) % 1.0
                r2, g2, b2 = colorsys.hls_to_rgb(complementary_h, 0.7, 0.8)
                wm_color = (int(r2 * 255), int(g2 * 255), int(b2 * 255))

        return wm_color

    def calculate_contrast_ratio(self, color1, color2):
        """计算两个颜色的对比度比例"""

        def get_luminance(color):
            """计算颜色的相对亮度"""
            r, g, b = color
            r = r / 255.0
            g = g / 255.0
            b = b / 255.0

            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4

            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        l1 = get_luminance(color1)
        l2 = get_luminance(color2)

        lighter = max(l1, l2)
        darker = min(l1, l2)

        return (lighter + 0.05) / (darker + 0.05)

    def create_watermark(self, original_watermark, watermark_color, target_size):
        """创建指定颜色和大小的水印"""
        # 调整大小
        resized_watermark = original_watermark.resize(target_size, Image.Resampling.LANCZOS)

        # 创建指定颜色的水印
        colored_watermark = Image.new("RGBA", target_size, (0, 0, 0, 0))

        # 获取alpha通道
        r, g, b, a = resized_watermark.split()

        # 创建新的RGB通道
        R = Image.new('L', target_size, watermark_color[0])
        G = Image.new('L', target_size, watermark_color[1])
        B = Image.new('L', target_size, watermark_color[2])

        # 应用透明度
        alpha = a.point(lambda p: p * self.opacity)

        # 合并通道
        colored_watermark = Image.merge('RGBA', (R, G, B, alpha))

        return colored_watermark

    def get_best_watermark_position(self, image, wm_width, wm_height):
        """智能选择最佳水印位置"""
        positions_to_try = []

        # 生成多个候选位置
        positions = [
            ('bottom-right', (image.width - wm_width - self.margin,
                              image.height - wm_height - self.margin)),
            ('bottom-left', (self.margin,
                             image.height - wm_height - self.margin)),
            ('top-right', (image.width - wm_width - self.margin, self.margin)),
            ('top-left', (self.margin, self.margin)),
        ]

        # 分析每个位置的背景
        best_position = None
        best_contrast = 0

        for pos_name, (x, y) in positions:
            bg_info = self.analyze_background_color(image, (x, y), wm_width, wm_height)

            # 计算对比度
            if bg_info['is_dark']:
                contrast_color = (255, 255, 255)  # 白色
            else:
                contrast_color = (0, 0, 0)  # 黑色

            contrast = self.calculate_contrast_ratio(bg_info['color'], contrast_color)

            # 考虑背景复杂度
            score = contrast * (1.0 - bg_info['complexity'] * 0.5)

            if score > best_contrast:
                best_contrast = score
                best_position = (x, y, bg_info)

        return best_position

    def resize_watermark(self, base_width, base_height):
        """根据原图大小调整水印尺寸"""
        # 计算目标尺寸
        target_width = int(base_width * self.scale)

        # 保持宽高比
        wm_ratio = self.watermark.width / self.watermark.height
        target_height = int(target_width / wm_ratio)

        # 确保最小尺寸
        min_size = 50
        if target_width < min_size or target_height < min_size:
            if target_width < target_height:
                target_width = min_size
                target_height = int(target_width / wm_ratio)
            else:
                target_height = min_size
                target_width = int(target_height * wm_ratio)

        return (target_width, target_height)

    def add_watermark(self, image_path, output_path=None):
        """为单张图片添加水印"""
        try:
            # 打开原始图片
            original_image = Image.open(image_path)
            original_format = original_image.format or 'JPEG'
            print("照片的原始格式是:", original_format)

            if original_format == 'MPO':
                # 获取图片数量
                original_image.seek(0)
                frame_count = 0
                try:
                    while True:
                        frame_count += 1
                        original_image.seek(original_image.tell() + 1)
                except EOFError:
                    pass

                # 读取所有图片
                images = []
                for i in range(frame_count):
                    original_image.seek(i)
                    images.append(original_image.copy())
                original_image = images[0]
                original_format = 'JPG'
                print(f"此MPO文件包含 {frame_count} 张图片，但是仅使用第一张，并设置格式为JPG")


            # 获取原始图片的EXIF数据
            original_exif = original_image.info.get('exif', b'')

            # 校正图片方向
            corrected_image = self.correct_image_orientation(original_image)

            new_exif_bytes = self.reset_exif_orientation(original_exif)

            # 转换为RGBA处理
            if corrected_image.mode != 'RGBA':
                if corrected_image.mode == 'P':  # 调色板模式
                    image = corrected_image.convert('RGBA')
                elif corrected_image.mode == 'LA':  # 灰度+透明
                    image = corrected_image.convert('RGBA')
                else:
                    image = corrected_image.convert('RGBA')
            else:
                image = corrected_image

            # 调整水印大小
            target_size = self.resize_watermark(image.width, image.height)

            # 分析背景并选择最佳位置
            if self.position == 'auto' and self.auto_color:
                # 智能选择最佳位置
                x, y, bg_info = self.get_best_watermark_position(image, target_size[0], target_size[1])
                position = (x, y)

                # 根据背景颜色选择水印颜色
                watermark_color = self.get_contrast_color(bg_info)

                print(f"  图片: {os.path.basename(image_path)}")
                print(f"    原始尺寸: {original_image.size} -> 校正后: {image.size}")
                print(f"    背景颜色: RGB{bg_info['color']}, 亮度: {bg_info['brightness']:.2f}")
                print(f"    选择颜色: RGB{watermark_color}")
                print(f"    位置: ({x}, {y})")
            else:
                # 使用指定位置
                position = self.calculate_position(image.width, image.height,
                                                   target_size[0], target_size[1])

                if self.auto_color:
                    # 分析指定位置的背景
                    bg_info = self.analyze_background_color(image, position,
                                                            target_size[0], target_size[1])
                    watermark_color = self.get_contrast_color(bg_info)

                    print(f"  图片: {os.path.basename(image_path)}")
                    print(f"    原始尺寸: {original_image.size} -> 校正后: {image.size}")
                    print(f"    背景颜色: RGB{bg_info['color']}, 亮度: {bg_info['brightness']:.2f}")
                    print(f"    选择颜色: RGB{watermark_color}")
                else:
                    # 使用固定颜色
                    watermark_color = (255, 255, 255)  # 白色
                    print(f"  图片: {os.path.basename(image_path)} - 使用白色水印")

            # 创建水印
            watermark = self.create_watermark(self.watermark, watermark_color, target_size)

            # 创建水印副本
            watermark_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
            watermark_layer.paste(watermark, position, watermark)

            # 合并水印
            watermarked = Image.alpha_composite(image, watermark_layer)

            # 保存图片
            if output_path is None:
                output_path = image_path

            # 根据原格式保存
            output_format = original_format.upper()

            # 保存图片，保留原始元数据
            if output_format in ['JPEG', 'JPG', 'JFIF']:
                # JPEG格式需要转换为RGB
                if watermarked.mode == 'RGBA':
                    # 创建白色背景
                    background = Image.new('RGB', watermarked.size, (255, 255, 255))
                    # 合并透明部分到白色背景
                    if watermarked.mode == 'RGBA':
                        background.paste(watermarked, mask=watermarked.split()[3])
                    watermarked = background

                # 保存时传递原始EXIF数据
                watermarked.save(output_path, "JPEG", quality=95, optimize=True, exif=new_exif_bytes)

            elif output_format == 'PNG':
                # PNG格式
                watermarked.save(output_path, "PNG", optimize=True)
            elif output_format == 'BMP':
                # BMP格式
                watermarked = watermarked.convert("RGB")
                watermarked.save(output_path, "BMP")
            elif output_format == 'TIFF' or output_format == 'TIF':
                # TIFF格式
                watermarked.save(output_path, "TIFF", compression='tiff_lzw')
            elif output_format == 'WEBP':
                # WEBP格式
                watermarked.save(output_path, "WEBP", quality=90, method=6)
            else:
                # 其他格式
                print("尝试保存的格式为:", output_format)
                watermarked.save(output_path, quality=95, optimize=True)

            return True, image_path

        except Exception as e:
            return False, f"{image_path}: {str(e)}"

    def process_folder(self, input_folder, output_folder=None,
                       extensions=None, recursive=True, skip_existing=True):
        """
        处理文件夹中的所有图片

        参数:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径 (None则覆盖原文件)
            extensions: 支持的图片扩展名
            recursive: 是否递归处理子文件夹
            skip_existing: 是否跳过已存在的文件
        """
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        # 收集所有图片文件
        image_files = []
        input_path = Path(input_folder)

        search_patterns = []
        for ext in extensions:
            # 添加小写和大写扩展名
            search_patterns.append(f"*{ext}")
            search_patterns.append(f"*{ext.upper()}")

        if recursive:
            for pattern in search_patterns:
                try:
                    found_files = list(input_path.rglob(pattern))
                    image_files.extend(found_files)
                except:
                    pass
        else:
            for pattern in search_patterns:
                try:
                    found_files = list(input_path.glob(pattern))
                    image_files.extend(found_files)
                except:
                    pass

        # 去重并转换为绝对路径
        image_files = list(set([str(f.absolute()) for f in image_files if f.is_file()]))

        if not image_files:
            print(f"在 {input_folder} 中未找到图片文件")
            return []

        # 处理输出路径
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None
            skip_existing = False  # 如果覆盖原文件，不需要跳过

        # 批量处理图片
        results = []
        skipped_count = 0

        for img_path_str in tqdm(image_files, desc=f"处理 {os.path.basename(input_folder)}"):
            img_path = Path(img_path_str)

            if output_path:
                # 保持原始目录结构
                try:
                    rel_path = Path(img_path_str).relative_to(input_path)
                    out_file = output_path / rel_path
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                except ValueError:
                    # 如果无法获取相对路径，使用文件名
                    out_file = output_path / img_path.name
            else:
                out_file = None

            # 检查是否跳过已存在的文件
            if skip_existing and out_file and out_file.exists():
                skipped_count += 1
                continue

            success, result = self.add_watermark(img_path_str, str(out_file) if out_file else None)
            results.append((success, result))

        if skip_existing and skipped_count > 0:
            print(f"  跳过了 {skipped_count} 个已存在的文件")

        return results

    def process_multiple_folders(self, input_folders, output_base=None,
                                 extensions=None, recursive=True, max_workers=1,
                                 skip_existing=True):
        """
        批量处理多个文件夹

        参数:
            input_folders: 输入文件夹列表
            output_base: 输出基础文件夹
            extensions: 支持的图片扩展名
            recursive: 是否递归处理子文件夹
            max_workers: 最大线程数
            skip_existing: 是否跳过已存在的文件
        """
        all_results = []

        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for input_folder in input_folders:
                input_folder_str = str(input_folder)
                if output_base:
                    folder_name = os.path.basename(input_folder_str)
                    output_folder = os.path.join(output_base, folder_name)
                else:
                    output_folder = None

                future = executor.submit(
                    self.process_folder,
                    input_folder_str,
                    output_folder,
                    extensions,
                    recursive,
                    skip_existing
                )
                futures.append((input_folder_str, future))

            # 收集结果
            for folder_name, future in futures:
                try:
                    results = future.result()
                    success_count = sum(1 for success, _ in results if success)
                    error_count = len(results) - success_count

                    print(f"\n文件夹 {os.path.basename(folder_name)}:")
                    print(f"  成功: {success_count}, 失败: {error_count}")

                    # 打印错误信息
                    for success, result in results:
                        if not success:
                            print(f"  错误: {result}")

                    all_results.extend(results)

                except Exception as e:
                    print(f"处理文件夹 {folder_name} 时出错: {str(e)}")

        return all_results


def main():
    """
    主函数 - 在这里直接配置参数
    """

    # ============== 配置参数 ==============

    # 1. 输入文件夹路径（支持多个文件夹）
    input_folders = [
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Zhangzhou",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Shenzhen",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Shanghai",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Huizhou",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/HKSAR",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Dongguan",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Shouguang",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Kunming",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Dali",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Lijiang",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Zhongshan",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Qingyuan",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Wuhan",
        "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/Zhucheng",
    ]

    # 2. 水印文件路径（必须是透明背景的PNG格式）
    watermark_path = "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/icon_w.png"

    # 3. 输出设置
    output_base = "D:/GitHub/willing-cui.github.io/images/gallery/_Camera/watermark"  # 输出基础文件夹
    # 如果设置为None，会覆盖原文件
    # output_base = None  # 使用这行会覆盖原文件

    # 4. 水印样式设置
    opacity = 0.7  # 水印透明度，0-1之间（0完全透明，1完全不透明）
    position = "auto"  # 水印位置: 'auto', 'bottom-right', 'bottom-left', 'top-right', 'top-left'
    scale = 0.15  # 水印大小相对于原图宽度的比例
    margin = 30  # 水印距离边缘的边距（像素）

    # 5. 颜色自适应设置
    auto_color = True  # 是否自动调整水印颜色
    min_contrast_ratio = 3.0  # 最小对比度比例（WCAG标准建议至少4.5:1，这里设为3.0）

    # 6. 处理选项
    recursive = True  # 是否递归处理子文件夹
    max_workers = 1  # 并行处理的线程数（设为1避免颜色分析混乱）
    skip_existing = True  # 跳过目标文件夹中已存在的文件
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}  # 支持的图片格式

    # ============== 参数验证 ==============

    # 验证水印文件是否存在
    if not os.path.exists(watermark_path):
        print(f"错误: 水印文件 {watermark_path} 不存在")
        print("请检查水印文件路径是否正确")
        return

    if not watermark_path.lower().endswith('.png'):
        print("警告: 水印文件应为PNG格式，否则可能无法正确显示透明背景")

    # 验证输入文件夹是否存在
    valid_folders = []
    for folder in input_folders:
        if os.path.exists(folder):
            valid_folders.append(folder)
        else:
            print(f"警告: 输入文件夹不存在，已跳过: {folder}")

    if not valid_folders:
        print("错误: 没有有效的输入文件夹")
        return

    # 检查输出文件夹中已存在的文件
    if output_base and skip_existing:
        print("\n检查目标文件夹中已存在的文件...")
        for folder in valid_folders:
            input_folder_name = os.path.basename(folder)
            output_folder = os.path.join(output_base, input_folder_name)

            if os.path.exists(output_folder):
                # 获取输出文件夹中的文件数量
                existing_files = []
                for ext in extensions:
                    for ext_case in [ext.lower(), ext.upper()]:
                        pattern = f"*{ext_case}"
                        try:
                            existing_files.extend(list(Path(output_folder).rglob(pattern)))
                        except:
                            pass

                if existing_files:
                    print(f"  {input_folder_name}: 已存在 {len(existing_files)} 个文件，将跳过这些文件")
                else:
                    print(f"  {input_folder_name}: 输出文件夹为空，将处理所有文件")

    # ============== 创建水印处理器 ==============

    print("\n正在初始化水印处理器...")
    print(f"水印文件: {watermark_path}")
    print(f"水印透明度: {opacity}")
    print(f"水印位置: {position}")
    print(f"水印比例: {scale}")
    print(f"水印边距: {margin}像素")
    print(f"颜色自适应: {'开启' if auto_color else '关闭'}")
    if auto_color:
        print(f"最小对比度: {min_contrast_ratio}:1")
    print(f"跳过已存在文件: {'是' if skip_existing and output_base else '否'}")

    try:
        processor = WatermarkProcessor(
            watermark_path=watermark_path,
            opacity=opacity,
            position=position,
            scale=scale,
            margin=margin,
            auto_color=auto_color,
            min_contrast_ratio=min_contrast_ratio
        )
    except Exception as e:
        print(f"初始化水印处理器失败: {str(e)}")
        return

    # ============== 批量处理 ==============

    print(f"\n开始处理 {len(valid_folders)} 个文件夹...")
    print(f"输出目录: {output_base if output_base else '覆盖原文件'}")
    print(f"处理模式: {'递归处理子文件夹' if recursive else '仅处理当前文件夹'}")
    print(f"跳过已存在文件: {'是' if skip_existing and output_base else '否'}")
    print("注意: 将自动校正图片方向，并保留原始EXIF元数据")

    # 如果设置了输出目录，确保它存在
    if output_base and not os.path.exists(output_base):
        os.makedirs(output_base, exist_ok=True)
        print(f"已创建输出目录: {output_base}")

    # 开始处理
    print("\n开始处理图片...")
    print("-" * 50)

    results = processor.process_multiple_folders(
        input_folders=valid_folders,
        output_base=output_base,
        extensions=extensions,
        recursive=recursive,
        max_workers=max_workers,
        skip_existing=skip_existing
    )

    # ============== 统计结果 ==============

    total_success = sum(1 for success, _ in results if success)
    total_failed = len(results) - total_success

    print(f"\n{'=' * 50}")
    print(f"处理完成!")
    print(f"处理了 {len(valid_folders)} 个文件夹")
    print(f"成功添加水印: {total_success} 张")
    print(f"失败: {total_failed} 张")

    if total_failed > 0:
        print("\n失败的文件:")
        for success, result in results:
            if not success:
                print(f"  {result}")

    if output_base:
        print(f"\n已保存水印图片到: {output_base}")
    else:
        print("\n已覆盖原文件添加水印")


if __name__ == "__main__":
    # 显示程序开始信息
    print("批量图片水印添加工具 - 智能颜色自适应版")
    print("=" * 60)
    print("功能说明:")
    print("- 自动校正图片方向，保持原始拍摄方向")
    print("- 自动分析水印位置的背景颜色")
    print("- 智能选择白色或黑色水印以获得最佳对比度")
    print("- 可选位置智能调整")
    print("- 支持透明度调节")
    print("- 批量处理多个文件夹")
    print("- 自动跳过已处理的文件")
    print("- 完整保留原始图片的EXIF元数据")
    print("=" * 60)

    # 运行主程序
    main()

    # 程序结束
    print("\n程序执行完毕")