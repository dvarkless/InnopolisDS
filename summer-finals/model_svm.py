import numpy as np
from image_feature_detector import (get_features, get_plain_data,
                                    threshold_high, threshold_low,
                                    threshold_mid)
from metrics import return_accuracy
from model_base import BaseModel


class SVM_Model(BaseModel):
    def __init__(self, data_converter=None, custom_params=None) -> None:
        super().__init__(data_converter, custom_params)

    def fit(self, data):
        y, x = self._splice_data(data)
        y = self.get_probabilities(y)
        self.params = np.zeros_like(x.T @ y)
        num_samples = x.shape[0]
        
        # забираем гиперпараметры, если есть
        learning_rate = getattr(self, 'learning_rate', 0.1)  
        epochs = getattr(self, "epochs", 100)

        for epoch in range(epochs):
            indexes = np.random.permutation(num_samples)
            y = y[indexes]
            x = x[indexes]
            for x_batch, y_batch in self._iter_batches(x, y):
                y_pred = self._forward(x_batch)
                self.params -= self._compute_gradient(
                    x_batch, y_pred, y_batch, learning_rate, k=1)

            loss = self.hinge_cost(x, y)

        return self

    def hinge_cost(self, x, y, k=1):
        loss = 1 - y * self._forward(x)
        loss[loss < 0] = 0

        loss = k * loss.mean(axis=0)

        cost = 0.5 * np.sum(self.params.T @ self.params, axis=0) + loss
        return cost

    def _compute_gradient(self, x, y_pred, y, lr=0.01, k=1) -> np.ndarray:
        loss = 1 - y * y_pred
        loss[loss <= 0] = 0
        loss[loss > 0] = 1
        positive_part = loss.mean(axis=0)

        diff = np.zeros_like(self.params)
        diff = self.params - (k * positive_part * (x.T @ y))

        return diff * lr

    def _forward(self, x):
        return x @ self.params

    def predict(self, x):
        self.data = x
        return self.get_labels(self._softmax(self.data @ self.params))


if __name__=='__main__':
    my_data = np.genfromtxt("datasets/light-train.csv",
                            delimiter=",", filling_values=0)
    test_data = np.genfromtxt(
        "datasets/light-test.csv", delimiter=",", filling_values=0)

    hp = {
        'num_classes': 26,
        'normalization': True,
        'shift_column': True,
    }
    num_classes = 26
    model = SVM_Model(data_converter=get_plain_data, custom_params=hp).fit(my_data)

    ans_test = model.predict(test_data[:, 1:])
    ans_train = model.predict(my_data[:, 1:])

    print(
        f'tran dataset accuracy = {return_accuracy(ans_train, my_data[:, 0])}')
    print(
        f'test dataset accuracy = {return_accuracy(ans_test, test_data[:, 0])}')
