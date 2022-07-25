import functools
import math
import time
from itertools import product

import numpy as np
from alive_progress import alive_bar


def timer(attr):
    @functools.wraps(attr)
    def _wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = attr(self, *args, **kwargs)
        runtime = time.perf_counter() - start
        if getattr(self, 'timer_on'):
            print(f'{runtime:.3f}s')
        return result
    return _wrapper


class ModelRunner:
    def __init__(self, model_class, timer=True, defaults=None, metrics=None) -> None:
        self.model_class = model_class
        self.timer_on = timer
        self.metrics = metrics
        self.metric_data = []
        self.parameters_data = []
        if defaults is not None:
            self.defaults = defaults

        self._curr_state = 'start'

    def run(self, train: np.ndarray, eval_input: np.ndarray, eval_ans: np.ndarray, params: dict, one_vs_one: bool = False):
        self.metric_data = []
        curr_params = dict()
        if one_vs_one:
            if len(list(params.values())) <= 1:
                pairs = list(*params.values())
            else:
                pairs = list(product(*list(params.values())))

            with alive_bar(len(list(pairs)), title=f'Проверка модели {self.model_class.__name__}', force_tty=True, bar='filling') as bar:
                for vals in pairs:
                    for i, key in enumerate(params.keys()):
                        try:
                            curr_params[key] = vals[i]
                        except TypeError:
                            curr_params[key] = vals

                    print('-----With parameters-----')
                    for key, val in curr_params.items():
                        print(f'{key} = {val}')

                    self.parameters_data.append(list(curr_params.values()))
                    self._run_method(train, eval_input, eval_ans, curr_params)
                    print('-----End with-----')
                    bar()
        else:
            iter_lens = [len(val) for val in params.values()]
            max_len = max(iter_lens)
            with alive_bar(max_len, title=f'Проверка модели {self.model_class.__name__}', force_tty=True, bar='filling') as bar:
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

                    self.parameters_data.append(list(curr_params.values()))
                    self._run_method(train, eval_input, eval_ans, curr_params)
                    bar()
                    print('-----End with-----')

        print("===============RESULTS=================")
        pos = self.highest_metric_pos(self.metric_data)
        print(f'On iteration {pos}:')
        print(f"With hyperparameters: {self.parameters_data[pos]}")
        print(f'Got metrics: {self.metric_data[pos]}')

    def _run_method(self, train: np.ndarray, eval_input: np.ndarray, eval_ans: np.ndarray, params: dict):
        params_to_pass = self.mix_params(self.defaults, params)
        self.model = self.model_class(params_to_pass)

        if self.timer_on:
            print('~fit complete in ', end='')
        self._run_train(train)

        if self.timer_on:
            print('~eval complete in ', end='')
        answer = self._run_eval(eval_input)
        self.comma_metrics(answer, eval_ans)

    def mix_params(self, main, invasive):
        maincpy = main.copy()
        for key, val in invasive.items():
            maincpy[key] = val
        return maincpy

    def comma_metrics(self, preds, evals):
        buff = []
        for metric in self.metrics:
            res = metric(preds, evals)
            print(f"    {metric.__name__} = {res:.3f}")
            buff.append(res)
        self.metric_data.append(buff)

    def highest_metric_pos(self, metrics):
        score = [math.prod(vals) for vals in metrics]
        return score.index(max(score))

    def get_metrics(self):
        return self.metric_data

    def get_params(self):
        return self.parameters_data

    @timer
    def _run_train(self, train: np.ndarray):
        self.model.fit(train)

    @timer
    def _run_eval(self, eval_input: np.ndarray):
        return self.model.predict(eval_input)
