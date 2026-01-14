from PIL import Image
import argparse
import sys


def replace_color_with_transparency(input_path, output_path, target_color, replacement_color=None, tolerance=30):
    """
    将图片中的特定颜色替换为透明，并可选地将剩余像素替换为另一种颜色

    参数:
    - input_path: 输入图片路径
    - output_path: 输出图片路径
    - target_color: 要替换为透明的目标颜色，格式为 (R, G, B) 或 (R, G, B, A)
    - replacement_color: 可选，替换剩余像素的颜色，格式为 (R, G, B) 或 (R, G, B, A)
    - tolerance: 颜色匹配容差（0-255），值越大匹配的颜色范围越广
    """
    try:
        # 打开图片并转换为RGBA模式（支持透明度）
        img = Image.open(input_path).convert("RGBA")
        pixels = img.load()

        # 确保目标颜色是4通道（RGBA）
        if len(target_color) == 3:
            target_color = (*target_color, 255)

        width, height = img.size

        # 遍历所有像素
        for x in range(width):
            for y in range(height):
                r, g, b, a = pixels[x, y]
                target_r, target_g, target_b, target_a = target_color

                # 计算与目标颜色的距离
                color_distance = ((r - target_r) ** 2 +
                                  (g - target_g) ** 2 +
                                  (b - target_b) ** 2) ** 0.5

                # 如果颜色在容差范围内，设为透明
                if color_distance <= tolerance:
                    pixels[x, y] = (r, g, b, 0)  # 设置alpha为0（完全透明）
                # 如果指定了替换颜色，替换剩余像素
                elif replacement_color is not None:
                    if len(replacement_color) == 3:
                        replacement_color_rgba = (*replacement_color, 255)
                    else:
                        replacement_color_rgba = replacement_color
                    pixels[x, y] = replacement_color_rgba

        # 保存为支持透明度的PNG格式
        if not output_path.lower().endswith('.png'):
            output_path = output_path.rsplit('.', 1)[0] + '.png'
            print(f"注意：输出格式自动转换为PNG以支持透明度: {output_path}")

        img.save(output_path, 'PNG')
        print(f"图片已保存: {output_path}")

        return True

    except Exception as e:
        print(f"处理图片时出错: {e}")
        return False


def rgb_string_to_tuple(rgb_str):
    """将'R,G,B'格式的字符串转换为颜色元组"""
    try:
        rgb_values = [int(x.strip()) for x in rgb_str.split(',')]
        if len(rgb_values) not in [3, 4]:
            raise ValueError("颜色格式应为 R,G,B 或 R,G,B,A")
        return tuple(rgb_values)
    except ValueError as e:
        print(f"颜色格式错误: {e}")
        sys.exit(1)


def main():
    success = replace_color_with_transparency(
        input_path="../icon_font.png",
        output_path="../icon_w.png",
        target_color=(255, 255, 255),  # 白色
        replacement_color=(255, 255, 255),  # 蓝色
        tolerance=30
    )

    if success:
        print("处理完成！")
    else:
        print("处理失败！")


if __name__ == "__main__":
    main()