import re
import numpy as np
import chardet 


def readPFM(file):
    file = open(file, 'rb')

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
    # 添加取绝对值和移除异常值
    data = np.abs(data)
    data[np.isinf(data)] = 0

    return data, scale
