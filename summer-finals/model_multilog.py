import numpy as np
from model_base import BaseModel, inval_check


class MultilogRegression(BaseModel):
    """
        Модель мультиноминальной логистической регрессии.

        Принцип работы:
            Обучающая выборка делится на мини выборки случайным образом, происходит расчет
            ответов по формуле:
                y[N, n_classes] = softmax(
                    X[N, n_input] @ theta[n_input, n_classes])
                где N - число значений в выборке,
                n_classes - число классов в ответе
                n_input - размерность входного вектора


            Далее происходит минимизация коэффициентов модели (model.params)
        с помощью градиентного спуска.

            parameters:
                Arbitrary:
                    data_converter - function-converter for input data
                    shift_column: bool - use a column of constants to act as additive
                                for folmula: wx_1 + wx_2 .... + wx_n + b = Y
                    normalization: bool - use normalization for input values
                    num_classes: int - number of classes in dataset
                    learning_rate: float - how fast model will fit
                    batch_size: int - size of mini-datasets to process while fitting.
                                      helps avoid overfitting
                    epochs: int - count of gradient descent steps for whole dataset
                    reg: str, function - regularization function (l1 or l2)
                Optional:
                    reg_w: float - regularization coefficient

    """

    def __init__(self, custom_params=None):
        self.must_have_params = ['data_converter', 'shift_column', 'normalization',
                                 'num_classes', 'learning_rate', 'batch_size', 'epochs',
                                 'reg']
        super().__init__(custom_params=custom_params)

    @inval_check
    def _compute_gradient(self, x, y_pred, y, lr=0.01, additive=None) -> np.ndarray:
        """
            Метод для расчета шага градиентного спуска.

            inputs:
                x      - input vector
                y_pred - model's predicted output
                y      - dataset's real answer for this x
                lr     - learning rate
                additive - add extra weight if necessary
                           (e.g. regularization)

            output:
                numpy.ndarray - gradient value
        """
        if additive is None:
            additive = 0
        data_size = x.shape[0]
        diff = y_pred - y
        # отключаем refcheck чтобы работал дебаггер
        diff.resize(data_size, y.shape[1], refcheck=False)
        gradient = (lr / data_size) * ((x.T @ diff) + additive)

        return gradient

    def _forward(self, x) -> np.ndarray:
        """
            Модель расчитывает ответ для заданного входа с использованием
            своей матрицы коэффициентов в данный момент

            input:
                x - input array
            output:
                y - output array, predicted answers
        """
        return self._softmax(x @ self.params)

    def fit(self, data: np.ndarray):
        # забираем гиперпараметры, если есть
        learning_rate = getattr(self, 'learning_rate', 0.01)
        # или оставляем дефолтные
        epochs = getattr(self, "epochs", 100)

        # Разбиваем датасет на входные данные и ответы
        y, x = self._splice_data(data)
        y = self.get_probabilities(y)
        num_samples = x.shape[0]

        if getattr(self, 'weight_init', 'zeros') == 'randn':
            self.params = np.random.randn(
                x.shape[1], getattr(self, 'num_classes')
            )
        else:
            self.params = np.zeros(
                (x.shape[1], getattr(self, 'num_classes'))
            )

        for epoch in range(epochs):
            # Разбиваем датасет на мини выборки
            indexes = np.random.permutation(num_samples)
            y = y[indexes]
            x = x[indexes]

            for x_batch, y_batch in self._iter_batches(x, y):
                y_pred = self._forward(x_batch)
                # Регуляризация
                reg_val = 0
                if getattr(self, 'reg', False):
                    reg_w = getattr(self, 'reg_w')
                    if getattr(self, 'reg', '') == 'l1':
                        reg_val = self.l2_reg(reg_w)
                    if getattr(self, 'reg', '') == 'l2':
                        reg_val = self.l2_reg(reg_w)
                # Градиентный спуск
                self.params -= self._compute_gradient(
                    x, y_pred, y_batch, lr=learning_rate, additive=reg_val
                )

            if hasattr(self, '_tick'):
                self._tick()

            if getattr(self, "debug", False):       # можно отображать коэффициенты на каждом шаге
                print(
                    f'-------------------------epoch {epoch}--------------------------')
                y_pred = self._forward(x)
                self.cost = self.BCEloss(
                    y_pred, y, num_samples, regularization=getattr(
                        self, "reg", None)
                ).sum()
                print(f"training cost: {self.cost}")
                print(f"params = {self.params}")
                print(f"_forward {self._forward(x)}")
                pred_labels = self.get_labels(self._softmax(y_pred))
                y_labels = self.get_labels(self._softmax(y))
                for metric in getattr(self, "metrics", []):
                    print(f"{metric.__name__} = {metric(pred_labels, y_labels)}")

        return self

    def predict(self, x):
        """
            Метод для предказания ответов для входного вектора,
            от model._forward отличается тем, что входные и
            выходные данные преобразовываются

            input:
                x - raw, non-normalized vector
            output:
                y - predicted labels
        """
        # Конвертируем значения
        self.data = x
        return self.get_labels(self._forward(self.data))
