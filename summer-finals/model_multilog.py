import matplotlib.pyplot as plt
import numpy as np

from image_feature_detector import get_features, get_plain_data
from metrics import return_accuracy
from model_base import BaseModel


class MultilogRegression(BaseModel):
    def __init__(self, input_size, out_size, data_converter=None, custom_params=None):
        super().__init__(data_converter=data_converter, custom_params=custom_params)
        input_size = data_converter(np.zeros(input_size)).shape[0]
        if self.weight_init == 'randn':
            self.params = np.random.randn(
                input_size, out_size
            )
        else:
            self.params = np.zeros(
                (input_size, out_size)
            )
        self.num_classes = out_size
        must_have_params = ['reg', 'metrics', 'reg_w', 'debug']
        self.assert_have(must_have_params)

    def _compute_gradient(self, x, y_pred, y, lr=0.01, additive=0):
        data_size = x.shape[0]
        diff = y_pred - y
        diff.resize(data_size, y.shape[1], refcheck=False)
        gradient = (lr / data_size) * ((x.T @ diff) + additive)

        return gradient

    def _softmax(self, x):
        mod_array = x + 2
        return (np.exp(mod_array).T / np.exp(mod_array).sum(axis=1)).T

    def get_params(self):
        return self.params

    def forward(self, x):
        return self._softmax(x @ self.params)

    def fit(self, data, learning_rate=0.01, batch_size=10, epochs=100):
        y, x = self._splice_data(data)
        y = self.get_probabilities(y)
        num_samples = x.shape[0]

        for epoch in range(epochs):
            indexes = np.random.permutation(num_samples)
            y = y[indexes]
            x = x[indexes]
            for i in range(int(x.shape[0] / batch_size)):
                curr_stack = i * batch_size
                x_batch = x[curr_stack: curr_stack + batch_size, :]
                y_batch = y[curr_stack: curr_stack + batch_size]

                y_pred = self.forward(x_batch)
                l2 = self.l2_reg(self.reg_w)
                self.params -= self._compute_gradient(
                    x, y_pred, y_batch, lr=learning_rate, additive=0
                )

            if self.debug:
                y_pred = self.forward(x)
                self.cost = self.BCEloss(
                    y_pred, y, num_samples, regularization=self.reg
                ).sum()
                print(f"training cost: {self.cost}")
                print(f"params = {self.params}")
                print(f"forward {self.forward(x)}")
                pred_labels = self.get_labels(self._softmax(y_pred))
                y_labels = self.get_labels(self._softmax(y))
                for metric in self.metrics:
                    print(f"{metric.__name__} = {metric(pred_labels, y_labels)}")

    def predict(self, x):
        self.data = x
        return self.get_labels(self.forward(self.data))


if __name__ == "__main__":
    my_data = np.genfromtxt("light-train.csv", delimiter=",", filling_values=0)
    num_classes = 26
    hp = {
        'weight_init': 'zeros',
        'reg': 'l2',
        'reg_w': 0.01,
        'debug': False,
        'metrics': [return_accuracy],
    }
    model = MultilogRegression(
        my_data.shape[1] - 1, num_classes, data_converter=get_plain_data, custom_params=hp
    )
    model.fit(my_data, learning_rate=0.02, batch_size=50, epochs=500)
    print("fit complete")
    ans = model.predict(my_data[:, 1:])
    print(return_accuracy(ans, my_data[:, 0]))
