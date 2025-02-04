import glob
import os


def find_png_files(directory):
    # 使用 glob 递归查找所有 PNG 文件
    return glob.glob(os.path.join(directory, '**', '*.png'), recursive=True)


def find_pfm_files(directory):
    # 使用 glob 递归查找所有 PNG 文件
    return glob.glob(os.path.join(directory, '**', '*.pfm'), recursive=True)


def save_to_txt(file_paths, output_file):
    with open(output_file, 'w') as f:
        for path in file_paths:
            f.write(path + '\n')


if __name__ == "__main__":
    # 指定要搜索的文件夹
    directory = '/8t/dataset/DrivingStereo/disp'

    # 指定输出文件路径
    output_file = '../KITTI2012_Driving/111.txt'

    # 查找所有 PNG 文件
    png_files = find_png_files(directory)
    png_files = sorted(png_files)

    # # 查找所有 PFM 文件
    # PFM_files = find_pfm_files(directory)
    # PFMfiles = sorted(PFM_files)

    # 将文件路径保存到 txt 文件中
    save_to_txt(png_files, output_file)

    print("找到 %d 个 image 文件，路径已保存到 %s" % (len(png_files), output_file))
