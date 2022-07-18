import math
import warnings

import numpy as np


class BaseModel:
    """
        Базовая модель.
        В ней определены вспомогательные методы, которые используются для обработки поступающих и исходящих данных. \n
        Также здесь определен общий для многих моделей функционал.

        use case:
            class MyModel(BaseModel):
                super.__init__(self, kwargs)

        inputs:
            data_converter: Function - функция-преобразователь для входных данных
            
            custom_params: dict - словарь с гиперпараметрами моделей, чтобы проверить 
                                  наличие обязательных параметров можно вызвать метод 
                                  self.must_have_params()

    """
    def __init__(self, data_converter=None, custom_params=None) -> None:
        self._data = np.array([])
        self.data_converter = data_converter if data_converter else lambda x: x
        self._params = None

        if custom_params:
            for key, val in custom_params.items():
                setattr(self, key, val)

    def assert_have(self, must_have_names: list) -> None:
        """
            Проверяет существует ли у класса аттрибуты с именами как в списке

        """
        for name in must_have_names:
            getattr(self, name)

    @property
    def data(self):    # Чтобы не вызывать функцию конвертации данных каждый раз
        return self._data   # когда нам надо конвертировать входные данные можно воспользоваться
                            # таким способом

    @data.setter
    def data(self, input_data):
        data = np.array(list(map(self.data_converter, input_data)))
        std = np.std(data, axis=0)
        std[std == 0] = 1
        self._data = (data - np.mean(data, axis=0)) / std   # Также здесь происходит нормализация

    @property
    def params(self):
        if math.isnan(self._params[0, 0]) or math.isinf(self._params[0, 0]):  # Проверяем происходило ли деление на ноль
            raise ValueError(f"self.params is invalid \n {self._params}")     # или переполнение переменной
        return self._params

    @params.setter
    def params(self, input_data):
        self._params = input_data

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, input_data):
        if not hasattr(self, '_cost_list'):  # Сохраняем значения стоимости чтобы вывести график, если понадобится
            self._cost_list = []
        self._cost_list.append(input_data)
        self._cost = input_data

    def l1_reg(self, weight: int) -> np.ndarray:      # L1 regularization
    """
        L1 regularization
        input - weight constant (0 < c < 1)
    """
        return weight * np.abs(self.params).sum(axis=0)

    def l2_reg(self, weight):      # L2 regularization
    """
        L2 regularization
        input - weight constant (0 < c < 1)
    """
        return weight * np.sqrt(np.power(self.params, 2)).sum(axis=0)

    def BCEloss(self, y_pred, y, sample_size, weight=0.01, regularization=None) -> np.ndarray:
        """
            Функция стоимости:
            Бинарная кросс энтропия 
            Используется в моделях, где выходные значения варьируются от 0 до 1.
            Отличается от математической формулы наличием ограничения для 
            минимального значения логарифма = -100 (как в Pytorch)

            inputs:
                y_pred - model's predicted values in range 0-1 (output)
                y      - true outputs in range 0-1
                sample size - batch size or dataset size
                weight - formula constant
                regularization: Function - pass the regularization function 
                         to add to the resulting formula
                regularization: str - pass 'l1' or 'l2' to use class methods for this job
        """
        log_val = np.log10(y_pred)
        np.putmask(log_val, log_val < -100, -100)
        add1 = y * log_val
        log_val = np.log10(1 - y_pred)
        np.putmask(log_val, log_val < -100, -100)
        add2 = (1 - y) * log_val
        out = -weight * (add1 + add2)
        out.resize(sample_size, out.shape[1], refcheck=False)

        if regularization == "l1":
            out += self.l1_reg(weight)
        elif regularization == "l2":
            out += self.l2_reg(weight)
        elif type(regularization) == type(self.l1_reg):
            out += regularization(weight)
        return out

    def get_probabilities(self, ans_arr: np.ndarray) -> np.ndarray:
        """
            Метод для получения вероятностей из ответов. 
            (отчет идет от 1)
            Пример:
                >> self.num_classes = 3
                >> a = [1, 2, 3]
                >> self.get_probabilities(a)
                [[1, 0, 0],
                 [0, 1, 0], 
                 [0, 0, 1]]
        """
        buff_arr = np.zeros(shape=(ans_arr.shape[0], self.num_classes))
        for i, val in enumerate(ans_arr):
            buff_arr[i, :] = self._ans_as_probs(val)
        return buff_arr

    def _ans_as_probs(self, ans):
        """
            Конвертирует каждое значение в вероятности
        """
        if hasattr(self, "num_classes"):
            out = np.zeros(self.num_classes)
            out[int(ans) - 1] = 1
        else:
            out = ans
            warnings.warn(
                """
                Отсутствует аттрибут BaseModel.num_classes,
                не получается конвертировать ответы в вероятности, возвращаю вход
                """
            )
        return out

    def get_labels(self, ans_arr):
        """
            Преобразует вероятности в ответы
            (отчет идет от 1)
            Пример:
                >> a = [[0.1, 0.5, 0.4],
                 [0.1, 0.1, 0.8], 
                 [1, 0, 0]]
                >> self.get_probabilities(a)
                 [2, 3, 1]

        """
        return np.array(list(map(self._ans_as_labels, ans_arr)))

    def _ans_as_labels(self, ans):
        """
            Конвертирует вероятности в значения
        """
        return np.argmax(ans) + 1

    def _splice_data(self, train_data) -> tuple:
        """
            Разделяет обучающий датасет EMNIST на ответы и входные значения

            также конвертирует и нормализует входные значения
            
            input: 
                train_data: np.ndarray - whole EMNIST dataset, first column is answers, 
                                         the rest is input
        """
        ans = train_data[:, 0]
        self.data = train_data[:, 1:]
        return ans, self.data

    def get_cost_list(self):
        """
            return costs per epoch if exists
        """
        try:
            return self._cost_list
        except AttributeError as e:
            print(e)
            raise AttributeError(
                'Чтобы получить график стоимости, надо сначала расчитать её: \nself.cost = self.cost_func(y_pred, y)  //пример')

    def predict(self, data):
        raise NotImplementedError