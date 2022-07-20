import numpy as np
from image_feature_detector import (get_features, get_plain_data,
                                    threshold_high, threshold_low,
                                    threshold_mid)
from metrics import return_accuracy
from model_base import BaseModel


class BayesianClassifier(BaseModel):
    """docstring for BayesianClassifier."""

    def __init__(self, input_size, out_size, data_converter=None, custom_params=None):
        super().__init__(data_converter=data_converter, custom_params=custom_params)
        input_size = data_converter(np.zeros(input_size)).shape[0]

    def fit(self, data):
        y, x = self._splice_data(data)
        y = self.get_probabilities(y)

        max_val = x.max()
        prob_in = x.T.mean(axis=1)/max_val
        prob_ans = y.mean(axis=0)

        prob_conditional = x.T @ y / y.sum(axis=0)
        prob_inverse = np.power(prob_in, -1)/10000
        prob_inverse[prob_inverse == np.inf] = 0
        self.params = np.outer(prob_inverse, prob_ans) * prob_conditional

        return self

    def predict(self, x):
        self.data = x
        return self.get_labels(self._softmax(self.data @ self.params))


if __name__ == "__main__":
    my_data = np.genfromtxt("datasets/light-train.csv",
                            delimiter=",", filling_values=0)
    test_data = np.genfromtxt(
        "datasets/light-test.csv", delimiter=",", filling_values=0)

    hp = {
        'num_classes': 26,
    }
    num_classes = 26
    model = BayesianClassifier(
        my_data.shape[1] - 1, num_classes, data_converter=threshold_mid, custom_params=hp).fit(my_data)

    ans_test = model.predict(test_data[:, 1:])
    ans_train = model.predict(my_data[:, 1:])

    print(
        f'tran dataset accuracy = {return_accuracy(ans_train, my_data[:, 0])}')
    print(
        f'test dataset accuracy = {return_accuracy(ans_test, test_data[:, 0])}')
