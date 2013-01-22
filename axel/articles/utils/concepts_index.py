"""
Building and maintaining index of words-concepts
This file is imported from settings
"""
from collections import defaultdict
from django.core.cache import cache

import sys


CONCEPT_PREFIX = 'concept:'
WORDS_SET = 'keywords'
EXPIRE = sys.maxint


def get_global_word_set():
    """
    Safely get global word set
    :rtype: set
    """
    if not cache.has_key(WORDS_SET):
        build_index()
    return cache.get(WORDS_SET)


def build_index():
    """
    Building index and putting it to cache
    Each concept will have a separate cache entry for the ease of update.
    Cache also
    """
    if cache.has_key(WORDS_SET):
        return
    from axel.stats.models import Collocations
    index = defaultdict(list)
    for id, concept in Collocations.objects.values_list('id', 'ngram'):
        for word in concept.split():
            index[word].append(id)
    for key, values in index.iteritems():
        cache.set(CONCEPT_PREFIX + key, values, EXPIRE)

    cache.set(WORDS_SET, set(index.keys()), EXPIRE)


def update_index(c_id, keywords):
    """
    Update index
    :type c_id: int
    :type keywords: unicode
    """
    if not cache.has_key(WORDS_SET):
        build_index()
    extra_words_set = set()
    for word in keywords.split():
        index = cache.get(word)
        if index:
            index.append(c_id)
            cache.set(CONCEPT_PREFIX + word, index, EXPIRE)
        else:
            extra_words_set.add(word)
            cache.set(CONCEPT_PREFIX + word, [c_id], EXPIRE)

    # If new words appeared - add them to global set
    if extra_words_set:
        global_set = cache.get(WORDS_SET)
        global_set.update(extra_words_set)
        cache.set(WORDS_SET, global_set, EXPIRE)
