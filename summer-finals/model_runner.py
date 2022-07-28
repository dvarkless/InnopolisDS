import functools
import math
import time
from itertools import product
from typing import Callable

import numpy as np
from alive_progress import alive_bar


def timer(attr):
    """
        Декоратор используется для вывода времени,
        за которое выполняется метод класса
    """
    @functools.wraps(attr)
    def _wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = attr(self, *args, **kwargs)
        runtime = time.perf_counter() - start
        print(f'{runtime:.3f}s')
        return result
    return _wrapper


class ModelRunner:
    """
        Класс предназначенный для удобного запуска моделей машинного обучения.

        Его возможности:
            - Создание экземпляров моделей с задаваемыми через словарь параметрами
              и их запуск через методы .fit() и .predict().
            - Вывод шкалы прогресса и времени выполнения методов моделей
            - Вывод различных метрик
            - Запуск одной модели с комбинацией различных параметров

        use case:
            >>> defaults = {'lr': 0.01, 'epochs': 100}
            >>> runner_inst = ModelRunner(ModelClass, timer=True, defaults=defaults, metrics=[accuracy])
            >>> runner_inst.run(training_data, eval_input, eval_answers, params={'lr': [0.001, 0.005], 'batch_size':[100],})

        inputs:
            model_class - Class of your model (not instance), all parameters should be passed through **kwargs
            defaults: dict - default kwargs for your model
            metrics: list - list of functions, they must take only two positional args: foo(preds, answers)
    """

    def __init__(self, model_class, defaults=None, metrics=None, responsive_bar=False) -> None:
        self.model_class = model_class
        self.metrics = metrics
        self._metric_data = []
        self._parameters_data = []
        if defaults is not None:
            self.defaults = defaults

        self._responsive_bar = responsive_bar

    def run(self, train: np.ndarray, eval_input: np.ndarray, eval_ans: np.ndarray, params: dict, one_vs_one: bool = False):
        """
            Запустить проверку моделей с заданными данными и параметрами.

            Итерируемые параметры задаются в словаре params в виде:
                >>> params = {
                >>>     'lr': [1,2,3,4]
                >>>     'epochs': [100, 200]
                >>>     }
            Количество шагов проверки при этом зависит также от способа сочетания
            параметров:
                - При one_vs_one=True все доступные параметры сочетаются друг
                с другом, в данном примере получается 8 шагов

                - При one_vs_one=False параметры берутся по столбцам, при этом
                если в каком то списке не хватает значений, то берется его последнее
                значение в списке. В данном примере получается 4 шага

            inputs:
                train - training dataset, first column is answer labels
                eval_input - evaluation dataset without answers
                eval_answers - answer array in the same order as eval_input
                               size = (1, N)
                params - dict consisted of lists of the iterated parameters.
                        every value must be a list, even singular vals
                one_vs_one - parameters combination method, True is One vertus One;
                            False is columswise combination.

        """
        self._metric_data = []
        self._models = []
        curr_params = dict()
        if one_vs_one:
            if len(list(params.values())) <= 1:
                pairs = list(*params.values())
            else:
                pairs = list(product(*list(params.values())))

            if self._responsive_bar:
                len_model_ticks = self.model_class(
                    self.defaults).define_tick(None, additive=len(eval_ans))
            else:
                len_model_ticks = 1
            with alive_bar(len(list(pairs)*len_model_ticks), title=f'Проверка модели {self.model_class.__name__}', force_tty=True, bar='filling') as bar:
                for vals in pairs:
                    for i, key in enumerate(params.keys()):
                        try:
                            curr_params[key] = vals[i]
                        except TypeError:
                            curr_params[key] = vals

                    print('-----With parameters-----')
                    for key, val in curr_params.items():
                        print(f'{key} = {val}')

                    self._parameters_data.append(list(curr_params.values()))
                    self._run_method(train, eval_input,
                                     eval_ans, curr_params, bar)
                    print('-----End with-----')
                    bar()
        else:
            iter_lens = [len(val) for val in params.values()]
            if self._responsive_bar:
                len_model_ticks = self.model_class(
                    self.defaults).define_tick(None, additive=len(eval_ans))
            else:
                len_model_ticks = 1
            max_len = max(iter_lens)
            with alive_bar(max_len*len_model_ticks, title=f'Проверка модели {self.model_class.__name__}', force_tty=True, bar='filling') as bar:
                for i in range(max_len):
                    for pos, key in enumerate(params.keys()):
                        this_len = iter_lens[pos]
                        try:
                            curr_params[key] = params[key][min(
                                this_len - 1, i)]
                        except TypeError:
                            curr_params[key] = params[key]

                    print('-----With parameters-----')
                    for key, val in curr_params.items():
                        print(f'{key} = {val}')

                    self._parameters_data.append(list(curr_params.values()))
                    self._run_method(train, eval_input,
                                     eval_ans, curr_params, bar)
                    bar()
                    print('-----End with-----')

        print("===============RESULTS=================")
        pos = self._highest_metric_pos(self._metric_data)
        print(f'On iteration {pos}:')
        print(f"With hyperparameters: {self._parameters_data[pos]}")
        print(f'Got metrics: {self._metric_data[pos]}')

    def _run_method(self, train: np.ndarray, eval_input: np.ndarray, eval_ans: np.ndarray, params: dict, bar_obj: Callable):
        """
            Внутренний обработчик ввода и вывода данных модели

            inputs:
                train - training dataset, first column is answer labels
                eval_input - evaluation dataset without answers
                eval_answers - answer array in the same order as eval_input
                               size = (1, N)
                params - dict of parameters that will be directly passed to the model


        """
        params_to_pass = self._mix_params(self.defaults, params)
        self.model = self.model_class(params_to_pass)

        if self._responsive_bar:
            self.model.define_tick(bar_obj, len(eval_ans))

        print('~fit complete in ', end='')
        self._run_train(train)

        print('~eval complete in ', end='')
        answer = self._run_eval(eval_input)
        self._comma_metrics(answer, eval_ans)
        self._models.append(self.model)

    def _mix_params(self, main, invasive):
        """
            Внутренний метод для изменения словаря с параметрами

            Вносит изменения в основной словарь с параметрами 
            из другого словаря. Основной словарь при этом не меняется.

            inputs:
                main: dict - dict to be  inserted values into
                invasive: dict - mixed in values
            output - new dict with mixed values
        """
        maincpy = main.copy()
        for key, val in invasive.items():
            maincpy[key] = val
        return maincpy

    def _comma_metrics(self, preds, evals):
        """
            Внутренний метод для получения метрик модели

            Можно в последствии получить все метрики через
            метод ModelRunner.get_metrics()

            inputs:
                preds: np.ndarray - model's predictions
                evals: np.ndarray - true labels
        """

        buff = []
        for metric in self.metrics:
            res = metric(preds, evals)
            print(f"    {metric.__name__} = {res:.3f}")
            buff.append(res)
        self._metric_data.append(buff)

    def _highest_metric_pos(self, metrics):
        """
            Внутренний метод для получения позиции
            наибольшего значения метрик.

            Если видов метрик больше 1, то сравнивается их
            среднее геометрическое.

            inputs: 
                metrics: list - list of metrics, list of lists or
                list of floats
            output - index of the biggest value
        """
        score = [math.prod(vals) for vals in metrics]
        return score.index(max(score))

    def get_models(self):
        return self._models

    def get_metrics(self):
        """
            Получить список со всеми значениями метрик

            Если метрик больше одной, то выдается список
            списков. Далее самим можно понять где какая метрика, 
            это не сложно

            output - list of all calculated metrics
        """
        return self._metric_data

    def get_params(self):
        """
            Получить список со всеми использованными
            гиперпараметрами

            Совпадает с тем, что передавалось в конструктор
            класса и в метод ModelRunner.run()

            output - list of hyperparameters
        """
        return self._parameters_data

    @timer
    def _run_train(self, train: np.ndarray):
        """
            Внутренний метод для запуска процесса 
            тренировки модели.

            inputs:
                train - training data
            decorator prints the time it takes to run
            this method
        """
        self.model.fit(train)

    @timer
    def _run_eval(self, eval_input: np.ndarray):
        """
            Внутренний метод для получения ответов модели.

            inputs:
                eval_input - data to process
            output: np.ndarray - model's predictions

            decorator prints the time it takes to run
            this method
        """

        return self.model.predict(eval_input)
