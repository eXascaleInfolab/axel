import time
import traceback

from django.conf import settings
import re


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


def _get_context(text, ngram, start=0):
    """
    Get first encountered context from text, full sentence
    :param ngram: n-gram to search for
    :param start: start of the ngram occurrence, optional
    :rtype: str
    """
    if start:
        word_start = start
    else:
        ngram = r's? '.join(ngram.split())+r's?'
        word_start = re.search(ngram, text)
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


def get_contexts_ngrams(text, ngram, bigger_ngrams):
    """
    GENERATOR
    Get all contexts from the text that do not contain bigger ngrams
    :returns: a pair (matched_ngram, context)
    :rtype: generator
    """
    # add possible plural forms
    ngram = r's? '.join(ngram.split())+r's?'
    for match in re.finditer(r'\s'+ngram+r'\s', text):
        context = _get_context(text, ngram, match.start() + 1)
        result = True
        for b_ngram in bigger_ngrams:
            if b_ngram in context:
                result = False
                break
        if result:
            yield match.group(0).strip(), context


def get_contexts(text, ngram, bigger_ngrams):
    """
    GENERATOR
    Get all contexts from the text that do not contain bigger ngrams,
    :returns: yields context for ngram without actual ngram (single/plural form)
    :rtype: generator
    """
    for matched_ngram, context in get_contexts_ngrams(text, ngram, bigger_ngrams):
        yield context


def print_progress(iterable, percent_step=1):
    total = float(len(iterable))
    abs_step = int((total * percent_step)/100) or 1
    for i, obj in enumerate(iterable):
        if i and not i % abs_step:
            print "{0:.2%} processed".format(i/total)
        yield obj
