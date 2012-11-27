"""
Building and maintaining index of words-concepts
This file is imported from settings
"""
from collections import defaultdict

from django.core.cache import cache
from axel.stats.models import Collocations

CONCEPT_PREFIX = 'concept:'
WORDS_SET = 'keywords'


def build_index():
    """
    Building index and putting it to cache
    Each concept will have a separate cache entry for the ease of update.
    Cache also
    """
    index = defaultdict(list)
    for id, concept in Collocations.objects.values_list('id', 'keywords'):
        for word in concept.split():
            index[word].append(id)
    for key, values in index.iteritems():
        cache.set(CONCEPT_PREFIX + key, values)

    cache.set(WORDS_SET, set(index.keys()))


def update_index(c_id, keywords):
    """
    Update index
    :type c_id: int
    :type keywords: unicode
    """
    extra_words_set = set()
    for word in keywords.split():
        index = cache.get(word)
        if index:
            index.append(c_id)
            cache.set(CONCEPT_PREFIX + word, index)
        else:
            extra_words_set.add(word)
            cache.set(CONCEPT_PREFIX + word, [c_id])

    # If new words appeared - add them to global set
    if extra_words_set:
        global_set = cache.get(WORDS_SET)
        global_set.update(extra_words_set)
        cache.set(WORDS_SET, global_set)
