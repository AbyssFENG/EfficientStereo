import os
import re


def get_unique_path(logs):
    # 如果路径不存在，直接返回
    if not os.path.exists(logs):
        return logs

    # 使用正则表达式检查路径末尾是否为数字
    match = re.search(r'(\d+)$', logs)

    if match:
        # 如果匹配到数字部分，提取并加1
        num = int(match.group(1))
        base_path = re.sub(r'(\d+)$', '', logs)  # 去掉末尾的数字
    else:
        # 未匹配数字则在结尾加一个0
        num = 0
        base_path = logs

    # 循环递增数字，直到找到一个不存在的路径
    while True:
        new_logs = "%s%s" % (base_path, num)
        if not os.path.exists(new_logs):
            break
        num += 1

    return new_logs


if __name__ == '__main__':
    # 示例使用
    logs = r"G:\NewStereo\logs\Light2D1"  # 你的logs路径
    unique_logs_path = get_unique_path(logs)
    print("logs 保存路径: ", unique_logs_path)
