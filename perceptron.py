class Perceptron:
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0

    def train(self, input_vecs, lables, iteration, rate):
        for _ in range(iteration):
            self._one_train(input_vecs, lables, rate)

    def _one_train(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (vec, lable) in samples:
            output = self.predict(vec)
            self._update_weights(vec, lable, output, rate)

    def predict(self, input_vec):
        parameters = zip(input_vec, self.weights)
        parameters = map(lambda x: x[0] * x[1], parameters)
        return self.activator(sum(parameters) + self.bias)

    def _update_weights(self, input_vec, label, output, rate):
        delta = label - output
        self.weights = list(map(lambda x: x[1] + rate * delta * x[0], zip(input_vec, self.weights)))
        self.bias += rate * delta

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return f'weights\t:{self.weights}\nbias\t:{self.bias}\n'


def f(x):
    return 1 if x > 0 else 0


def get_train_dataset():
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_predict():
    p = Perceptron(2, f)
    input_vecs, labels = get_train_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perception = train_and_predict()
    print(and_perception)
    print(f'1 and 1 = {and_perception.predict([1,1])}')
    print(f'1 and 0 = {and_perception.predict([1,0])}')
    print(f'0 and 1 = {and_perception.predict([0,1])}')
    print(f'0 and 0 = {and_perception.predict([0,0])}')