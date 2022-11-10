import functools
import time
import torch

def no_grad(attr):
    """
    Убирает расчет градиентов для тензоров в целом методе
    """
    @functools.wraps(attr)
    def _wrapper(self, *args, **kwargs):
        with torch.no_grad():
            result = attr(self, *args, **kwargs)
        return result
    return _wrapper


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
        self.last_runtime = runtime
        return result
    return _wrapper
