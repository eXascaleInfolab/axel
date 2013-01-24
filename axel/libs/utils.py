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


def get_context(text, ngram, start=0):
    """
    Get first encountered context from text, full sentence
    :param ngram: n-gram to search for
    :param start: start of the ngram occurrence, optional
    :rtype: str
    """
    if start:
        word_start = start
    else:
        word_start = text.find(ngram, start)
    # Check possible punctuations
    context_start = 0
    context_end = len(text)
    for punct in ('.','?',';'):
        punct_rpos = text.rfind(punct, 0, word_start)
        punct_lpos = text.find(punct, word_start)
        if punct_rpos != -1:
            context_start = max(punct_rpos + 1, context_start)
        if punct_lpos != -1:
            context_end = min(punct_lpos + 1, context_end)
    return text[context_start:context_end].strip()


def get_contexts(text, ngram):
    """
    Get all contexts from the text
    :rtype: list
    """
    contexts = []
    word_start = text.find(ngram, 0)
    while word_start != -1:
        contexts.append(get_context(text, ngram, word_start))
        word_start = text.find(ngram, word_start + 1)
    return contexts


def print_progress(iterable, percent_step=1):
    total = float(len(iterable))
    abs_step = int((total * percent_step)/100) or 1
    for i, obj in enumerate(iterable):
        if i and not i % abs_step:
            print "{0:.2%} processed".format(i/total)
        yield obj
