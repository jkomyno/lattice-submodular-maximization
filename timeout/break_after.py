import time
import signal
from typing import Callable, Any, Iterator, TypeVar, Union

T = TypeVar('T')


class TimeoutException(Exception):   # Custom exception class
    pass


def break_after(seconds=2):
    def timeout_handler(signum, frame):   # Custom signal handler
        raise TimeoutException

    def function(fn: Callable[[Any], Iterator[T]]):
        def wrapper(*args, **kwargs) -> Union[None, T]:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            result = None

            try:
                for tmp_result in fn(*args, **kwargs):
                    result = tmp_result

                # clear alarm
                signal.alarm(0)
            except TimeoutException:
                print(u'Timeout: %s sec reached.' % seconds, fn.__name__, args, kwargs)

            return result
        return wrapper

    return function
