"""Part-of-speech calculation"""
from collections import defaultdict
import nltk

from axel.libs.nlp import Stemmer


def compress_pos_tag(max_ngram, rules_dict):
    """compress POS ngram tag
    :param rules_dict: correspondence rules on how to compress tags,
    values should contain compiled regular expressions, for example {'PLURAL': ('.*NNS.*', ... )}
    :type rules_dict: list
    """
    for key, regex in rules_dict:
        if regex.search(max_ngram):
            return key
    return max_ngram


def pos_tag_pos(ngram, contexts, tag_pos=-1):
    """
    Identifies POS tag for the ngram in each context and returns the corresponding dict with counts
    :type ngram: unicode
    :type contexts: list
    :rtype: dict
    When ngram is right at the beginning of the sentence, this code actually takes the last (-1)
    POS tag, which happens to be a punctuation mark.
    """
    ngram_tags = defaultdict(lambda: 0)
    if not contexts:
        contexts = [(ngram, ngram)]
    ngram_len = len(ngram.split())
    if tag_pos > 0:
        tag_pos += ngram_len - 1
    for i, context in enumerate(contexts):
        words, context = context
        words = tuple(words.split())
        tag = None
        tags = [(word, tag)
                for word, tag in nltk.pos_tag(nltk.regexp_tokenize(context, Stemmer.TOKENIZE_REGEXP))]
        for j, wordtag in enumerate(tags):
            if wordtag[0] == words[0] and tuple(zip(*tags)[0][j:j+ngram_len]) == words:
                try:
                    tag = tags[j + tag_pos][1]
                except IndexError:
                    pass
                break
        if tag:
            ngram_tags[tag] += 1

    return ngram_tags.items()


def pos_tag(ngram, contexts):
    """
    Identifies POS tag for the ngram in each context and returns the MAX probable
    :type ngram: unicode
    :type contexts: list
    :rtype: list
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

    # select max weight combination
    return ngram_tags.items()
