import os
import numpy as np
from PIL import Image
import chardet
import re


# 读取 PFM 文件的函数
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    encode_type = chardet.detect(header)
    header = header.decode(encode_type['encoding'])
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encode_type['encoding']))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode(encode_type['encoding']))
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    data = np.abs(data)
    data[np.isinf(data)] = 0

    return data, scale


def save_numpy_as_png(numpy_array, output_path):
    """
    将 NumPy 数组保存为 PNG 文件。
    如果数组是浮点数类型，直接将其转换为 16 位整数。
    """
    # 如果数组是浮点数类型，转换为 16 位整数
    if numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
        numpy_array = (numpy_array * 255).astype(np.uint16)  # 将浮点数映射到 0-255 范围
    elif numpy_array.dtype == np.uint8:
        numpy_array = numpy_array.astype(np.uint16) * 256  # 将 8 位整数转换为 16 位整数

    # 将 NumPy 数组保存为 PNG 文件
    if len(numpy_array.shape) == 2:  # 单通道图像
        img = Image.fromarray(numpy_array, mode='I;16')  # 16 位灰度图像
    elif len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:  # RGB 图像
        img = Image.fromarray(numpy_array, mode='RGB')  # RGB 图像
    else:
        raise ValueError("Unsupported array shape for PNG conversion.")

    img.save(output_path)


def convert_pfm_to_png(folder_path):
    """
    遍历文件夹中的所有 PFM 文件，将其转换为 PNG 文件。
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是 PFM 文件
        if filename.endswith('.pfm'):
            # 构建完整的文件路径
            pfm_file_path = os.path.join(folder_path, filename)

            # 读取 PFM 文件
            numpy_array, _ = readPFM(pfm_file_path)

            # 构建输出 PNG 文件的路径
            png_filename = filename.replace('.pfm', '.png')
            png_file_path = os.path.join(folder_path, png_filename)

            # 将 NumPy 数组保存为 PNG 文件
            save_numpy_as_png(numpy_array, png_file_path)
            print("Converted: %s to %s" % (filename, png_filename))


# 示例用法
folder_path = r'F:\数据集\Middle\disp'  # 替换为你的文件夹路径
convert_pfm_to_png(folder_path)