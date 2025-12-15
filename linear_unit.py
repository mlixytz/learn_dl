from perceptron import Perceptron

import json


def f(x): return x


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        super().__init__(input_num, f)


def get_training_dataset():
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5300, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    lu = LinearUnit(1)
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu


if __name__ == '__main__':
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    # 测试
    print(f'Work 3.4 years, monthly salary = {linear_unit.predict([3.4]):.2f}')
    print(f'Work 15 years, monthly salary = {linear_unit.predict([15]):.2f}')
    print(f'Work 1.5 years, monthly salary = {linear_unit.predict([1.5]):.2f}')
    print(f'Work 6.3 years, monthly salary = {linear_unit.predict([6.3]):.2f}')
