import time
import traceback

from django.conf import settings


def print_timing(func):
    """
    Decorator to print function execution time.
    Also takes stack information into account to perform pretty_print
    """
    if settings.DEBUG:
        def print_timing_wrapper(*arg, **kwargs):
            time1 = time.time()
            result = func(*arg, **kwargs)
            time2 = time.time()
            res = '%s: %0.3fms' % (func.func_name, (time2 - time1) * 1000.0)
            call_stack = zip(*traceback.extract_stack())[2]
            tab_init = '    ' * (call_stack.count('print_timing_wrapper')-1)
            print tab_init, res
            return result
        return print_timing_wrapper
    else:
        return func
