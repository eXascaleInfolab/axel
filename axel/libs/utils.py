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
        ngram = r's? '.join(ngram.split()) + r's?'
        word_start = re.search(ngram, text, re.I).start()
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
    def _ngram_plural_regex(ngram):
        """
        :type ngram: unicode
        """
        ngram = ur''.join([x if x in (' ', '-', '') else
                           ur'({0}|{1})({2}|{3})(s|es)?'.format(x[0], x[0].upper(), x[1:],
                            x[1:-1]+ur'ies') for x in re.split(r'([\s\-])', ngram)])
        return ur'(?:[^\w\-]|^)(?P<orig>{0})(?:[^\w\-]|$)'.format(ngram)

    regex_ngram = _ngram_plural_regex(ngram)
    skip_count = 0
    for match in re.finditer(regex_ngram, text, re.U):
        if skip_count:
            skip_count -= 1
            continue
        context = _get_context(text, ngram, match.start('orig'))
        # we need to keep ngram count and in the end set the skip count number correctly,
        # because in one sentence there can be multiple occurrences.
        ngram_count = len(re.findall(regex_ngram, context, re.U)) or 1
        b_ngram_count = 0
        result = True
        for b_ngram in bigger_ngrams:
            b_ngram_count += len(re.findall(_ngram_plural_regex(b_ngram), context, re.U))
            if b_ngram_count == ngram_count:
                result = False
                break
        if result:
            skip_count = b_ngram_count
            orig_ngram = match.group('orig').strip()
            # lower case first letter if it's not title nor acronym
            if not orig_ngram.istitle() and orig_ngram[0].isupper() and \
                not orig_ngram.split()[0].isupper():
                orig_ngram2 = orig_ngram[0].lower() + orig_ngram[1:]
                context = context.replace(orig_ngram, orig_ngram2)
                orig_ngram = orig_ngram2
            yield orig_ngram, context


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
