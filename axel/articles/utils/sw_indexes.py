"""ScienceWISE-specific scores calculation"""
import pickle

from django.core.cache import cache

import sys


_WORD_COUNTS_PREFIX = 'SW_word_counts'
_EXPIRE = sys.maxint


def _get_global_word_counts():
    """
    Build global SW word set and put it to cache if does not exist
    It is used in partial word-concept matching
    :rtype: dict
    """
    if not cache.has_key(_WORD_COUNTS_PREFIX):
        # file contains dict of words with counts from SW ontology, we will use it directly
        word_counts = pickle.load(open('word_counts.pcl'))
        cache.set(_WORD_COUNTS_PREFIX, word_counts, _EXPIRE)
        return word_counts
    return cache.get(_WORD_COUNTS_PREFIX)


def get_concept_score(concept):
    """
    :param concept: concept to get score for
    :type concept: unicode
    :rtype: int
    """
    word_counts = _get_global_word_counts()
    return sum([word_counts[c] for c in set(concept.split()).intersection(word_counts)])
