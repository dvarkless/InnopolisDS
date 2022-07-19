import numpy as np
from image_feature_detector import get_features, get_plain_data
from metrics import return_accuracy
from model_base import BaseModel


class MultilogRegression(BaseModel):
    """
        Модель мультиноминальной логистической регрессии.

        Принцип работы:
            Обучающая выборка делится на батчи случайным образом, происходит расчет
            ответов по формуле:
                y[N, n_classes] = softmax(X[N, n_input] @ theta[n_input, n_classes])
                где N - число значений в выборке,
                n_classes - число классов в ответе
                n_input - размерность входного вектора


            Далее происходит минимизация коэффициентов модели (model.params) 
            с помощью градиентного спуска.

            inputs:
                input_size - input vector's length for one sample
                out_size   - output vector's length for one sample
                data_converter - look in parent's class (BaseModel)
                custom_params - custom parameters for this model:
                                must_have_params = ['reg', 'metrics', 'reg_w', 'debug']
                                possible_params = ['weight_init']
    """

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
        # must_have_params = ['metrics','debug']
        # self.assert_have(must_have_params)

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

    def _softmax(self, x) ->np.ndarray:
        """
            Возвращает мягкий максимум для массива.
            Сумма значений по одному измерению равна 1,
            таким образом можно трактовать полученный массив
            как массив векторов вероятностей

            Вычисляется по строкам.

            input:
                x - input array
            output:
                numpy.ndarray
        """
        mod_array = x + 2   # Таким образом можно избежать деления на крохотные числа
        return (np.exp(mod_array).T / np.exp(mod_array).sum(axis=1)).T

    def get_params(self) -> np.ndarray:
        return self.params

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
        """
            Метод разделеняет входную выборку на подвыборки и 
            находит оптимальные параметры для расчета.

            inputs:
                data - input data, answers + input vectors
                       [:, 0] - answers
                       [:, 1:] - input vectors
                learning_rate: float
                batch_size - subsample size
                epochs - number of iterations for whole sample
                         number of gradient steps:(epochs*(N/batch_size))

            output:
                self - model instance
        """
        learning_rate = getattr(self, 'learning_rate', 0.01)  # забираем гиперпараметры, если есть
        batch_size = getattr(self, "batch_size", 10)          # или оставляем дефолтные
        epochs = getattr(self, "epochs", 100)

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
                y_pred = self._forward(x_batch)

                reg_val = 0
                if getattr(self, 'reg', False):
                    reg_w = getattr(self, 'reg_w')
                    if getattr(self, 'reg', '')=='l1':
                        reg_val = self.l2_reg(reg_w)
                    if getattr(self, 'reg', '')=='l2':
                        reg_val = self.l2_reg(reg_w)

                self.params -= self._compute_gradient(
                    x, y_pred, y_batch, lr=learning_rate, additive=reg_val
                )

            if getattr(self, "debug", False):       # можно отображать коэффициенты на каждом шаге
                print(f'-------------------------epoch {epoch}--------------------------')
                y_pred = self._forward(x)
                self.cost = self.BCEloss(
                    y_pred, y, num_samples, regularization=getattr(self, "reg", None)
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
        self.data = x
        return self.get_labels(self._forward(self.data))


if __name__ == "__main__":
    my_data = np.genfromtxt("datasets/light-train.csv",
                            delimiter=",", filling_values=0)
    num_classes = 26
    hp = {
        'learning_rate': 0.02,
        'batch_size': 50,
        'epochs': 400,
        'weight_init': 'zeros',
        'normalization': True,
        'reg': 'l2',
        'reg_w': 0.01,
        'debug': False,
        'metrics': [return_accuracy],
    }

    test_data = np.genfromtxt(
            "datasets/light-test.csv", delimiter=",", filling_values=0)
    for param in ['None', 'l1', 'l2']:
        hp['reg'] = param
        print(f'-----{param}-----')
        model = MultilogRegression(
            my_data.shape[1] - 1, num_classes, data_converter=get_plain_data, custom_params=hp
        ).fit(my_data)

        ans_test = model.predict(test_data[:, 1:])
        ans_train = model.predict(my_data[:, 1:])

        print(f'tran dataset accuracy = {return_accuracy(ans_train, my_data[:, 0])}')
        print(f'test dataset accuracy = {return_accuracy(ans_test, test_data[:, 0])}')
    
