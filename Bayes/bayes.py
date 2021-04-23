# encoding: utf-8

import numpy as np
import struct
import matplotlib.pyplot as plt
from load_data import *


def PCA(X, k=50):
    print(X.shape)
    assert len(X.shape) == 2, 'Wrong shape of input'
    X_mean = np.mean(X, axis=0)
    X_diff_mean = X - X_mean
    X_cov = X_diff_mean.T.dot(X_diff_mean) / (X.shape[0] - 1)

    eig_vals, eig_vecs = np.linalg.eig(X_cov)

    # eig_vecs[:, np.where(eig_vals < 0)] = - eig_vecs[:, np.where(eig_vals < 0) ]
    # eig_vals = np.abs(eig_vals)

    sort_index = np.argsort(- np.abs(eig_vals))
    eig_vals_sorted = eig_vals[sort_index]

    total_eig_val = sum(eig_vals)
    var_ratio = eig_vals_sorted / total_eig_val
    # print('var:{}\n sum ratio:{}'.format(eig_vals_sorted[:k], var_ratio[:k].sum())

    sort_index = sort_index[:k]
    eig_vals_sorted = eig_vals[sort_index]
    eig_vecs_sorted = eig_vecs[:, sort_index]

    return X_diff_mean.dot(eig_vecs_sorted)


def parameter_estimation(data, label):
    """
    图像数据 1行784列，每列的数据为正态分布
    因此对每一维的均值和方差进行估计
    :param data: n*784的图像数据，未分标签
    :param label: 图像标签
    :return: 10类784列的均值和方差
    """
    print("Estimate params")
    label_count = [0] * 10
    for element in label:
        label_count[int(element)] += 1
    # print(label_count)

    # 先验概率
    probability_prior = np.zeros(10)
    for i in range(10):
        probability_prior[i] = label_count[i] / 60000

    # 高斯分布的参数
    # use a LIST store 10 np ARRAY
    data_processed = [0] * 10
    for i in range(0, 10):
        data_processed[i] = np.zeros((label_count[i], len(data[0])), dtype=complex)
        # print(data_processed[i].shape)

    # catagorize data, traverse all train data length=60000
    label_flag = [0] * 10  # mark current position to fill
    for i in range(0, len(data)):
        i_label = int(label[i])
        data_processed[i_label][label_flag[i_label]] = data[i]
        label_flag[i_label] += 1
    # print(label_flag)
    # for i in range(10):
    #     print(i)
    #     print(data_processed[i].shape)

    # to estimate all parameters MEAN and VAR
    # calculate by col
    # result like 10 * 2 * 784
    parameter_est = [[[(col + 1) * (row + 1) * (j + 1) for col in range(784)] for row in range(2)] for j in range(10)]

    for i in range(10):
        cur = data_processed[i]
        parameter_est[i][0] = np.mean(cur, axis=0)
        parameter_est[i][1] = np.sqrt(np.var(cur, axis=0))
        # print(parameter_est[i][0][60:74])

    return parameter_est, probability_prior


def predict(param, probability_prior, data):
    """
    predict catagorize 计算每个点的概率密度和先验概率之积
    :param param: 由训练样本估计的均值方差
    :param probability_prior: 模式类先验概率
    :param data: 测试用的数据
    :return: 预测结果 数据类型np.array
    """
    probs = []
    for i in range(10):
        temp_mu, temp_var = param[i][0], param[i][1]
        temp_var = temp_var * temp_var
        prob = 1 * np.exp(- np.power((data - temp_mu), 2) / (2 * temp_var)) / np.sqrt(2 * np.pi * temp_var)
        probs.append(np.prod(prob, axis=1))

    probs_v = np.vstack(probs)
    pred = np.argmax(probs_v, axis=0)
    print(pred.shape)
    return pred


def accuracy(res, label):
    res = np.array(res)
    label = np.array(label)
    # print(res)
    # print(label[0:len(res)])
    print(sum(res == label[0:len(res)]) / len(res))


def main():
    data = load_train_images()
    label = load_train_labels()
    data_test = load_test_images()
    label_test = load_test_labels()
    data_total = np.vstack((data, data_test))

    data_total_pca = PCA(data_total)
    data = data_total_pca[:data.shape[0]]
    data_test = data_total_pca[data.shape[0]:]
    # print(data_test[0])
    param, probability_prior = parameter_estimation(data, label)

    # for i in range(10):
    #     plt.imshow(np.array(param[i][0]).reshape(28, 28), cmap='gray')
    #     plt.show()

    res = predict(param, probability_prior, data_test[0:10000])

    accuracy(res, label_test)


if __name__ == '__main__':
    main()
