from __future__ import division
from collections import Counter, defaultdict
import re
from axel.articles.models import Article
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
    else:
        local_ngrams = set(build_ngram_index(ngram).keys()).intersection(article_dict.keys())
        # select biggest prev combination
        if local_ngrams:

            score = sum([article_dict[ngram][1] for ngram in local_ngrams])
            # reduce score from the consumed ngram according its score
        # no - full average
        else:
            score = weight_both_ngram_score(ngram, text, article_dict)
        return score


def populate_article_dict(queryset, score_func):
    """
    :type queryset: QuerySet
    """
    article_dict = defaultdict(dict)
    rel_ngram_set = set(queryset.filter(tags__is_relevant=True))
    irrel_ngram_set = set(queryset.filter(tags__is_relevant=False))
    for article in Article.objects.filter(cluster_id=queryset.model.CLUSTER_ID)[:20]:
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
            score = score_func(ngram.ngram, text, article_dict[article])
            article_dict[article][ngram.ngram] = (ngram_abs_count, score, is_rel)
    return article_dict
