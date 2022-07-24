import numpy as np
from image_feature_detector import get_features, get_plain_data, threshold_mid
from metrics import return_accuracy
from model_base import BaseModel


class BayesianClassifier(BaseModel):
    """
        Модель Наивного Байесовского классификатора.

        Использует теорему байеса для определения класса точки в пространстве:
            P(Y | a0, a1, ... an) = (P(Y) * P(a0, a1, ... an | Y)) / P(a0, a1, ... an)
            P() - вероятность
            Y - ответ модели (0 или 1)
            a0, a1, ... an - параметры модели

        Для мультиклассовой классификации в данном случае значение Y представлено в виде
        вектора Y = (Y1, Y2... Ym). В качестве ответа выбирается наибольшая вероятность в
        векторе.

        Для корректного расчета вероятностей параметров, входные значения должны быть равны
        0 или 1
    """

    def __init__(self, custom_params=None):
        super().__init__(custom_params)

    def fit(self, data):
        y, x = self._splice_data(data)
        y = self.get_probabilities(y)

        max_val = x.max()
        prob_in = x.T.mean(axis=1)/max_val  # P(a0, a1, ... an)
        prob_ans = y.mean(axis=0)  # P(Y)

        prob_conditional = x.T @ y / y.sum(axis=0)  # P(a0, a1, ... an | Y)
        # Получаем обратную матрицу. Уменьшаем максимальное значение, чтоб работал softmax
        prob_inverse = np.power(prob_in, -1)/10000
        # Избавляемся от деления на 0
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
        'data_converter': get_plain_data,
        'num_classes': 26,
    }
    num_classes = 26
    model = BayesianClassifier(custom_params=hp).fit(my_data)

    ans_test = model.predict(test_data[:, 1:])
    ans_train = model.predict(my_data[:, 1:])

    print(
        f'tran dataset accuracy = {return_accuracy(ans_train, my_data[:, 0])}')
    print(
        f'test dataset accuracy = {return_accuracy(ans_test, test_data[:, 0])}')
