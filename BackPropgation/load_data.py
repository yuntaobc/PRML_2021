# encoding: utf-8

import numpy as np
import struct
import matplotlib.pyplot as plt

# 训练集文件
train_images_idx3_ubyte_file = '../MNIST/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = '../MNIST/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = '../MNIST/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = '../MNIST/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为magic数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'  # '>IIII'是说使用大端法读取4个unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # print("offset: ", offset)
    fmt_image = '>' + str(image_size) + 'B'  # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, image_size))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape(image_size)
        offset += struct.calcsize(fmt_image)
    # print(images.shape)
    # print(images[1])
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        # if (i + 1) % 10000 == 0:
            # print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def run():
    train_images = load_train_images()  # (num_rows*num_cols,num_images)
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    print(train_images.shape)
    print(train_labels.shape)


    # 查看前十个数据及其标签以读取是否正确
    # for i in range(10):
    #     print(train_labels[i])
    #     print(train_images[i])
    #     a = np.array(train_images[i])
    #     plt.imshow(a.reshape(28,28), cmap='gray')
    #     plt.show()
    print('done')


if __name__ == '__main__':
    run()
