"""Part-of-speech calculation"""
from collections import defaultdict
import nltk
import re

from axel.articles.utils.nlp import Stemmer


def compress_pos_tag(max_ngram, rules_dict):
    """compress POS ngram tag
    :param rules_dict: correspondence rules on how to compress tags,
    values should contain compiled regular expressions, for example {'PLURAL': ('.*NNS.*', ... )}
    :type rules_dict: dict
    """
    for key, regex in rules_dict.iteritems():
        if regex.search(max_ngram):
            return key
    return max_ngram


def pos_tag_prev(ngram, contexts):
    """
    Identifies POS tag for the ngram in each context and returns the MAX probable
    :type ngram: unicode
    :type contexts: list
    :rtype: unicode
    """
    ngram_tags = defaultdict(lambda: 0)
    if not contexts:
        contexts = [(ngram, ngram)]
    ngram_len = len(ngram.split())
    for i, context in enumerate(contexts):
        words, context = context
        words = tuple(words.split())
        tag = None
        tags = [(word, tag)
                for word, tag in nltk.pos_tag(nltk.regexp_tokenize(context, Stemmer.TOKENIZE_REGEXP))]
        for j, wordtag in enumerate(tags):
            if wordtag[0] == words[0] and tuple(zip(*tags)[0][j:j+ngram_len]) == words:
                tag = tags[j-1][1]
                break
        if tag:
            ngram_tags[tag] += 1

    return ngram_tags


def pos_tag(ngram, contexts):
    """
    Identifies POS tag for the ngram in each context and returns the MAX probable
    :type ngram: unicode
    :type contexts: list
    :rtype: unicode
    """
    ngram_tags = defaultdict(lambda: 0)
    if not contexts:
        contexts = [(ngram, ngram)]
    ngram_len = len(ngram.split())
    for i, context in enumerate(contexts):
        words, context = context
        words = tuple(words.split())
        tags = [(word, tag)
                for word, tag in nltk.pos_tag(nltk.regexp_tokenize(context, Stemmer.TOKENIZE_REGEXP))
                if word in set(words)]
        for j, wordtag in enumerate(tags):
            if wordtag[0] == words[0] and tuple(zip(*tags)[0][j:j+ngram_len]) == words:
                tags = tuple(zip(*tags)[1][j:j+ngram_len])
                break

        ngram_tags[tags] += 1
        # check every 10 iterations for speed
        if not i % 10 and i > 1:
            if len(ngram_tags) == 1:
                break
            else:
                items = sorted(ngram_tags.items(), key=lambda x: x[1], reverse=True)
                if items[0][1] > 5*items[1][1]:
                    break

    # select max weight combination
    max_ngram_tags = max(ngram_tags.items(), key=lambda x: x[1])[0]
    #max_ngram = _compress_pos_tag(max_ngram_tags)
    #if max_ngram == 'XXX VERB' and obj.is_relevant:
    #    print ngram_tags, ngram.ngram
    #if max_ngram in ('JJ NN','NN NN') and not obj.is_relevant:
    #    print 'IRREL:', ngram_tags, ngram.ngram
    #if 'PRP$' in max_ngram:
    #    print ngram
    return ' '.join(max_ngram_tags)
