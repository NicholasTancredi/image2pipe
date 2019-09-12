from typing import Dict, List, Callable, Optional, Any
from time import time
from functools import partial, wraps

import numpy as np
from prettytable import PrettyTable


Times = Dict[str, List[float]]


class Timer:
    times: Dict[str, List[float]]
    _tics: Dict[str, float]
    title: str

    def __init__(self, times: Optional[Times] = None, title: str = 'Timer'):
        self.title = title
        if times is not None:
            self.times = times
        else:
            self.times = {}
        self._tics = {}

    def tic(self, name: str, time_value: Optional[float] = None) -> Callable:
        """
            Returns a name bound toc method for convenience
            Optionall set the time yourself
        """
        if name in self._tics:
            print(f'WARNING! Trying to override a tic without calling toc first on name {name}... automatically performing toc for you.')
            self.toc(name)

        if time_value is not None:
            self._tics[name] = time_value
        else:
            self._tics[name] = time()

        return partial(self.toc, name=name)

    def toc(self, name: str, time_value: Optional[float] = None) -> float:
        if name not in self.times:
            self.times[name] = []

        if name not in self._tics:
            raise Exception(f'Must first call "tic" on this name {name}')

        if time_value is not None:
            current_time = time_value
        else:
            current_time = time()

        result = current_time - self._tics.pop(name)

        self.times[name].append(result)

        return result

    def decorator(self, func: Callable) -> Any:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = func.__name__
            toc = self.tic(name)
            result = func(*args, **kwargs)
            toc()
            return result
        return wrapper

    def __getitem__(self, key) -> Callable:
        # this allows you to not have to call tic, but ultimately its pretty pointless.
        return self.tic(key)

    def print(self, show_details: bool = False):
        if len(self.times) == 0:
            print('Nothing was set on Timer!')
            return

        x = PrettyTable()
        if show_details:
            x.field_names = ['Name', 'Total', 'Average', 'Count', 'std', 'max', 'min']
        else:
            x.field_names = ['Name', 'Total', 'Average', 'Count']

        x.align["Name"] = "l"
        for key, value in self.times.items():
            if len(value) > 0:
                if show_details:
                    x.add_row([key, round(np.sum(value), 4), round(np.mean(value), 4), len(value), round(np.std(value), 4), round(np.max(value), 4), round(np.min(value), 4)])
                else:
                    x.add_row([key, round(np.sum(value), 4), round(np.mean(value), 4), len(value)])
        print('\n')
        print(self.title)
        print(x)

    def merge_times(self, new_times: Times):
        for key, value in new_times.items():
            if key not in self.times:
                self.times[key] = []
            self.times[key] += value


def combine_timers(*timers: Timer) -> Timer:
    """
        Helper method if you need to combine timers together into one
    """
    times: Times = {}
    for timer in timers:
        for key, value in timer.times.items():
            if key not in times:
                times[key] = []

            times[key] += value
    return Timer(times)


def main():
    """
        Example Usage
    """
    from time import sleep
    timer = Timer()
    for i in range(3):
        # most basic
        toc = timer.tic('test')
        sleep(0.1)
        toc()

        # alternative basic
        timer.tic('alt_test')
        sleep(0.1)
        timer.toc('alt_test')

        # you can use __getitem__ syntax as well (same as calling tic)
        toc = timer['using_set_item_example']
        sleep(0.1)
        toc()

        # decorator usage
        @timer.decorator
        def func():
            sleep(0.1)

        func()

        # if you want you can also pass the time info down
        toc = timer.tic('pass_time_info', time_value=time())
        sleep(0.1)
        toc(time_value=time())

    # combine timers together if you need
    timer2 = Timer()

    timer2.merge_times(timer.times)

    toc = timer2.tic('timer2')
    sleep(0.1)
    toc()
    timer = combine_timers(timer, timer2, Timer(), Timer())
    # simple output
    timer.print()
    # detailed output
    timer.print(show_details=True)


if __name__ == '__main__':
    main()
