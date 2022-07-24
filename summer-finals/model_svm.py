import numpy as np
from image_feature_detector import PCA_transform, get_features, get_plain_data
from metrics import return_accuracy
from model_base import BaseModel


class SVM_Model(BaseModel):
    """ 
        Модель Машины опорных векторов для мультиклассовой классификации.

        Строит гиперплоскость с максимальным расстоянием от точек в пространстве,
        разделяющую входные данные на 2 класса.

        Для классификации на 3 и более класса надо построить несколько моделей
        и сравнить их результаты способом 'один против одного' или 
        'один против остальных' (One vs One, One vs Rest).
        На данный момент реализован способ One vs Rest

        Применяется функция стоимости Soft Margin Loss:
        https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2#66a2

        Для работы функции стоимости надо ответы на тренировочной выборке
        подавать в виде: [-1, -1, 1, -1],  где позиция единицы - номер класса точки.

    """

    def __init__(self, custom_params=None) -> None:
        super().__init__(custom_params)

    def fit(self, data):
        y, x = self._splice_data(data)
        y = self._ovr_split(y)
        self.params = np.zeros_like(x.T @ y)
        num_samples = x.shape[0]

        # забираем гиперпараметры, если есть
        learning_rate = getattr(self, 'learning_rate', 0.1)
        epochs = getattr(self, "epochs", 100)
        k = getattr(self, "regularization", 1)
        debug = getattr(self, "debug", False)

        for epoch in range(epochs):
            indexes = np.random.permutation(num_samples)
            y = y[indexes]
            x = x[indexes]
            for x_batch, y_batch in self._iter_batches(x, y):
                y_pred = self._forward(x_batch)
                self.params -= self._compute_gradient(
                    x_batch, y_pred, y_batch, learning_rate, k=k)

            loss = self.soft_margin_loss(self._forward(x), y)

            if debug:
                print(f'-----epoch {epoch+1}/{epochs}-----  ', end='')
                print(f'Current cost: {loss.sum():.2f}')

        if debug:
            print(
                f'''params properties: mean={self.params.mean()},  
                std={self.params.std()}, max={self.params.max()}, 
                min={self.params.min()}
                ''')

            return self

    def hinge_loss(self, y_pred, y):
        """
            Применяется формула:
            max(0, 1 - y_i*(w*x_i+b)) - для каждого значения в массиве

            inputs:
                y_pred - model's output
                y - training data answer

            output:
                np.array (shape = n_classes)
        """
        return np.vectorize(lambda x: x if x > 0 else 0)(1-y*y_pred).mean(axis=0)

    def soft_margin_loss(self, y_pred, y, k=1):
        """
            Формула:
            J = k * ||w||^2 + 1/N * Sum(max(0, 1 - y_i*(w*x_i+b)))

            inputs:
                y_pred - model's output
                y - training data answer
                k - regularization strength

            output:
                np.array (shape = n_classes)
        """
        return (self.hinge_loss(y_pred, y) + (k * self.params.T @ self.params)).mean(axis=0)

    def _ovr_choose(self, x):
        """
            Возвращаем позицию наибольшего значения

            input:
                x - input array, 2 dims

            output - np.ndarray, 1 dim
        """
        return np.argmax(x, axis=1) + 1

    def _ovr_split(self, x):
        """
            Получаем единицу на позиции значения ответа, как в 
            .get_probabilities(), только остальные значения 
            приравниваются к -1.

            input:
                x - input array, 2 dims

            output - np.ndarray, 1 dim
        """
        return self.get_probabilities(x, custom_fun=self._plus_minus_one)

    def _plus_minus_one(self, x):
        """
            Конвертирует каждое значение в 1 или -1
        """
        num = getattr(self, "num_classes")
        out = np.zeros(num) - 1
        out[int(x) - 1] = 1
        return out

    def _compute_gradient(self, x, y_pred, y, lr=0.01, k=1) -> np.ndarray:
        diff = np.zeros_like(self.params)
        diff = self.params - (k * self.soft_margin_loss(y_pred, y) * (x.T @ y))

        return diff * lr

    def _forward(self, x):
        return x @ self.params

    def predict(self, x):
        self.data = x
        return self._ovr_choose(self.data @ self.params)


if __name__ == '__main__':
    my_data = np.genfromtxt("datasets/light-train.csv",
                            delimiter=",", filling_values=0)
    test_data = np.genfromtxt(
        "datasets/light-test.csv", delimiter=",", filling_values=0)

    hp = {
        'data_converter': get_plain_data,
        'num_classes': 26,
        'epochs': 200,
        'batch_size': 10,
        'learning_rate': 0.0001,
        'regularization': 0.01,
        'normalization': True,
        'shift_column': True,
        'debug': True,

    }
    num_classes = 26

    pca_transformer = PCA_transform(100).fit(my_data)
    model = SVM_Model(custom_params=hp).fit(my_data)

    ans_test = model.predict(test_data[:, 1:])
    ans_train = model.predict(my_data[:, 1:])

    print(
        f'tran dataset accuracy = {return_accuracy(ans_train, my_data[:, 0])}')
    print(
        f'test dataset accuracy = {return_accuracy(ans_test, test_data[:, 0])}')
