from __future__ import division
from collections import Counter, defaultdict
import re
from axel.articles.models import Article
from axel.articles.utils.nlp import build_ngram_index

NGRAM_REGEX = ur'(?:\w|-)'


def weight_both_ngram(ngram, text, article_dict):
    """AVERAGE BETWEEN TWO SCORES"""
    score = 0
    space_index = -1
    denominator = 0
    while True:
        space_index = ngram.find(' ', space_index + 1)
        if space_index == -1:
            break
        w1, w2 = ngram[:space_index], ngram[space_index + 1:]
        distribution_dict = Counter(re.findall(ur'({1}+) {0}'.format(w2, NGRAM_REGEX, re.U), text))
        N2 = sum(distribution_dict.values())
        N2_len = len(distribution_dict)
        score += distribution_dict[w1] / N2
        distribution_dict = Counter(re.findall(ur'{0} ({1}+)'.format(w1, NGRAM_REGEX, re.U), text))
        N1 = sum(distribution_dict.values())
        N1_len = len(distribution_dict)
        score += distribution_dict[w2] / N1
        if (N1_len == 1 or N2_len == 1) and text.count(ngram) > 5:
            score += 2
        denominator += 2
    return score / denominator


def weight_both_ngram1(ngram, text, article_dict):
    """AVERAGE BETWEEN TWO WEIGHTED SCORES"""
    score = 0
    space_index = -1
    denominator = 0
    while True:
        space_index = ngram.find(' ', space_index + 1)
        if space_index == -1:
            break
        w1, w2 = ngram[:space_index], ngram[space_index + 1:]
        distribution_dict = Counter(re.findall(ur'({1}+) {0}'.format(w2, NGRAM_REGEX, re.U), text))
        N1 = sum(distribution_dict.values())
        N1_len = len(distribution_dict)
        score += distribution_dict[w1]
        distribution_dict = Counter(re.findall(ur'{0} ({1}+)'.format(w1, NGRAM_REGEX, re.U), text))
        N2 = sum(distribution_dict.values())
        N2_len = len(distribution_dict)
        score += distribution_dict[w2]
        score /= (N1 + N2)
        denominator += 1
    return score / denominator


def weight_both_ngram2(ngram, text, article_dict):
    """MODIFIER DETERMINER"""
    score = 0
    space_index = -1
    denominator = 0
    while True:
        space_index = ngram.find(' ', space_index + 1)
        if space_index == -1:
            break
        w1, w2 = ngram[:space_index], ngram[space_index + 1:]
        distribution_dict = Counter(re.findall(ur'({1}+) {0}'.format(w2, NGRAM_REGEX), text, re.U))
        N2 = sum(distribution_dict.values())
        N2_len = len(distribution_dict)
        score1 = distribution_dict[w1]
        distribution_dict = Counter(re.findall(ur'{0} ({1}+)'.format(w1, NGRAM_REGEX), text, re.U))
        N1 = sum(distribution_dict.values())
        N1_len = len(distribution_dict)
        score2 = distribution_dict[w2]
        score = (score1 + score2) / (N1 + N2)
        # 4% increase in MAP
        if N2_len / N1_len >= 5:
            score /= 2
        elif N1_len / N2_len >= 5:
            score += 1
        denominator += 1
    return score / denominator


def weight_ngram_score(ngram, text, article_dict, ngram_abs_count):
    if len(ngram.split()) == 2:
        return weight_both_ngram2(ngram, text, article_dict)
    else:
        smaller_ngrams = set(build_ngram_index(ngram).keys()).intersection(article_dict.keys())
        # select max split combination
        if smaller_ngrams:
            if len(smaller_ngrams) == 1:
                smaller_ngram = smaller_ngrams.pop()
                smaller_ngram_count, smaller_ngram_score, is_rel = article_dict[smaller_ngram]
                score = smaller_ngram_score * ngram_abs_count / smaller_ngram_count
                # reduce score from the consumed ngram according its score
                article_dict[smaller_ngram] = (smaller_ngram_count,
                                               smaller_ngram_score - score, is_rel)
            else:
                smaller_ngrams = sorted(smaller_ngrams, key=lambda x: len(x.split()), reverse=True)
                smaller_ngram_count, smaller_ngram_score, is_rel = article_dict[smaller_ngrams[0]]
                score = smaller_ngram_score * ngram_abs_count / smaller_ngram_count
                article_dict[smaller_ngrams[0]] = (smaller_ngram_count,
                                                   smaller_ngram_score - score, is_rel)
        # no - full average
        else:
            score = weight_both_ngram2(ngram, text, article_dict)
        return score


def populate_article_dict(queryset, score_func):
    """
    :type queryset: QuerySet
    """
    article_dict = defaultdict(dict)
    rel_ngram_set = set(queryset.filter(tags__is_relevant=True))
    irrel_ngram_set = set(queryset.filter(tags__is_relevant=False))
    for article in Article.objects.filter(cluster_id=queryset.model.CLUSTER_ID):
        text = article.stemmed_text
        for ngram in sorted(article.articlecollocation_set.all(),
                            key=lambda x: len(x.ngram.split())):
            if ngram.ngram in rel_ngram_set:
                is_rel = True
            elif ngram.ngram in irrel_ngram_set:
                is_rel = False
            else:
                continue
            ngram_abs_count = text.count(ngram.ngram)
            if ngram_abs_count <= 5:
                continue
            score = score_func(ngram.ngram, text, article_dict[article], ngram_abs_count)
            article_dict[article][ngram.ngram] = (ngram_abs_count, score, is_rel)
    return article_dict
