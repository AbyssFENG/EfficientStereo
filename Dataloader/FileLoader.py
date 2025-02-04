def load_paths_from_txt(file_path):
    paths = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉每行的换行符并添加到列表
            paths.append(line.strip())
    return paths


def TrainLoader(filepath):

    left_txt = filepath + 'PartDrivingStereoLeft.txt'
    left_train = load_paths_from_txt(left_txt)
    right_txt = filepath + 'PartDrivingStereoRight.txt'
    right_train = load_paths_from_txt(right_txt)
    disp_txt = filepath + 'PartDrivingStereoDisp.txt'
    disp_train = load_paths_from_txt(disp_txt)

    return left_train, right_train, disp_train


def TestLoader(filepath):

    left_val_txt = filepath + 'PartTestDrivingStereoLeft.txt'
    left_val = load_paths_from_txt(left_val_txt)
    right_val_txt = filepath + 'PartTestDrivingStereoRight.txt'
    right_val = load_paths_from_txt(right_val_txt)
    disp_val_txt = filepath + 'PartTestDrivingStereoDisp.txt'
    disp_val = load_paths_from_txt(disp_val_txt)

    return left_val, right_val, disp_val


def ValLoader(filepath):

    left_val_txt = filepath + 'PartTestDrivingStereoLeft.txt'
    left_val = load_paths_from_txt(left_val_txt)
    right_val_txt = filepath + 'PartTestDrivingStereoRight.txt'
    right_val = load_paths_from_txt(right_val_txt)

    return left_val, right_val