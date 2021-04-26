from layer import *
from load_data import *


# 随机初始化参数
# 获取输入数据并分批次
# 计算梯度更新样本
# 进行测试，统计准确率
# ref：https://blog.csdn.net/weixin_43496455/article/details/103075072


def main():
    # get data
    # no PCA
    train_images = load_train_images()  # (num_images, num_rows*num_cols)
    train_labels = load_train_labels()  # (num_images, 10)

    # test_images = load_test_images()
    # test_labels = load_test_labels()

    train_account = train_images.shape[0]
    nodes_input = train_images.shape[1]
    print(f"Sample account: {train_account}. node_input: {nodes_input}.")

    # omit layer_input because it contains no calculate
    layer_middle = Layer(input_dim=nodes_input, output_dim=50)
    layer_output = Layer(input_dim=50, output_dim=10)

    learning_rate = 0.1

    # train start
    # use mini_batch, generate random array index
    batch_all = batch_generate(train_account, batch_size=50)
    for batch in batch_all:
        # 前向传播
        images = train_images[batch]
        labels = train_labels[batch]

        layer_middle_res = layer_middle.call(inputs=images)
        layer_output_res = layer_output.call(inputs=layer_middle_res)
        # output = np.argmax(layer_output_res, axis=1)

        # 反向传播更新参数
        # output layer to middle layer
        sum = []
        diff = layer_output_res * (1 - layer_output_res) * (labels - layer_output_res)
        for g, b in zip(diff, layer_middle_res):
            sum.append(np.outer(g, b))
        sum = np.array(sum)

        sum_current = np.array([np.outer(g, b) for g, b in zip(diff, layer_middle_res)])
        sum_current = np.array(sum_current)

        print(sum==sum_current)

        delta_w_output = np.mean(sum, axis=0) * learning_rate
        delta_b_output = np.mean(diff, axis=0) * learning_rate * (-1)  # 50x10 axis=0 calculate mean by col

        layer_output.update(delta_w_output, delta_b_output)

        # middle layer to input layer



def batch_generate(sample_account, batch_size):
    """
    generate mini batch randomly
    :param sample_account:
    :return: n group of array index to represent random batch
    """
    batch_account = int(sample_account / batch_size)
    result = np.zeros(shape=(batch_account, batch_size), dtype=np.int32)

    # TODO: realize real random array
    for i in range(batch_account):
        result[i] = range(i * batch_size, i * batch_size + batch_size, 1)

    return result


if __name__ == '__main__':
    main()
    # batch = batch_generate(6000, 28)
    # num = np.random.rand(200) * 100
    # print(num[batch[0]])
    pass
