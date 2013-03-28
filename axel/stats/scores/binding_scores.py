from collections import Counter
import re
from axel.articles.utils.nlp import build_ngram_index

NGRAM_REGEX = ur'(?:\w|-)'


def weight_both_ngram_score(ngram, text, article_dict):
    score = 0
    space_index = -1
    denominator = 0
    while True:
        space_index = ngram.find(' ', space_index + 1)
        if space_index == -1:
            break
        w1, w2 = ngram[:space_index], ngram[space_index + 1:]
        distribution_dict = Counter(re.findall(ur'({1}+) {0}'.format(w2, NGRAM_REGEX), text))
        N1 = sum(distribution_dict.values())
        N1_len = len(distribution_dict)
        if N1 == 0:
            return score
        score += distribution_dict[w1] / N1
        distribution_dict = Counter(re.findall(ur'{0} ({1}+)'.format(w1, NGRAM_REGEX), text))
        N2 = sum(distribution_dict.values())
        N2_len = len(distribution_dict)
        if N2 == 0:
            return score
        score += distribution_dict[w2] / N2
        if (N1_len == 1 or N2_len == 1) and text.count(ngram) > 5:
            score += 2
        denominator += 2
    return score / denominator


def weight_ngram_score(ngram, text, article_dict):
    if len(ngram.split()) == 2:
        return weight_both_ngram_score(ngram, text, article_dict)
    elif len(ngram.split()) == 3:
        local_ngrams = set(build_ngram_index(ngram).keys()).intersection(article_dict.keys())
        # check bigram inside
        if local_ngrams:
            score = sum([article_dict[ngram][1] for ngram in local_ngrams])
        # no - full average
        else:
            score = weight_both_ngram_score(ngram, text, article_dict)
        return score
