"""Collection scorings calculation"""
import pickle
import sys

from django.core.cache import cache
from django.conf import settings
from axel.libs.nlp import build_ngram_index


_WORD_COUNTS_PREFIX = 'SW_word_counts'
_NGRAM_COUNTS_PREFIX = 'SW_ngram_counts'
_EXPIRE = sys.maxint
ontology = set([x for y in pickle.load(open(settings.ABS_PATH('ontology.pcl'))).values() for x in y])

def _get_global_word_counts():
    """
    Build global SW word set and put it to cache if does not exist
    It is used in partial word-concept matching
    :rtype: dict
    """
    if not cache.has_key(_WORD_COUNTS_PREFIX):
        # file contains dict of words with counts from SW ontology, we will use it directly
        word_counts = pickle.load(open(settings.ABS_PATH('word_counts.pcl')))
        cache.set(_WORD_COUNTS_PREFIX, word_counts, _EXPIRE)
        return word_counts
    return cache.get(_WORD_COUNTS_PREFIX)


def _get_global_ngram_counts():
    """
    Build global SW ngram set and put it to cache if does not exist
    It is used in full ngram-concept matching
    :rtype: dict
    """
    if not cache.has_key(_NGRAM_COUNTS_PREFIX):
        # file contains dict of words with counts from SW ontology, we will use it directly
        ngram_counts = pickle.load(open(settings.ABS_PATH('counts.pcl')))
        cache.set(_NGRAM_COUNTS_PREFIX, ngram_counts, _EXPIRE)
        return ngram_counts
    return cache.get(_NGRAM_COUNTS_PREFIX)


def get_word_concept_score(ngram):
    """
    :param ngram: ngram to get score for
    :type ngram: unicode
    :rtype: int
    """
    word_counts = _get_global_word_counts()
    return sum([word_counts.get(c, 0) for c in set(ngram.split())])


def get_ngram_concept_score(ngram):
    """
    :param ngram: ngram to get score for
    :type ngram: unicode
    :rtype: int
    """
    ngram_counts = _get_global_ngram_counts()
    if ngram in ngram_counts:
        del ngram_counts[ngram]
    return u','.join(ngram_counts.keys()).count(ngram)


def get_concept_ngram_score(ngram):
    """
    :param ngram: ngram to get score for
    :type ngram: unicode
    :rtype: int
    """
    ngram_counts = _get_global_ngram_counts()
    if ngram in ngram_counts:
        del ngram_counts[ngram]
    return len(set(build_ngram_index(ngram).keys()).intersection(ngram_counts.keys()))

# import POS tagging scorings
from .postag import *
# import ACM DL search scores
from .dl_acm_search import *
