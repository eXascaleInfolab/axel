from __future__ import division
from collections import Counter, defaultdict, OrderedDict
import re
from axel.articles.models import Article
from axel.libs.nlp import build_ngram_index
import nltk

NGRAM_REGEX = ur'(?:\w|-)'


def abs_count_score(collection_ngram, ngram, text, article_dict, ngram_abs_count, *args, **kwargs):
    """
    :type collection_ngram: Collocation
    :type ngram: ArticleCollocation
    :type ngram_abs_count: int
    """
    return ngram_abs_count, {}, {}


def rel_count_score(collection_ngram, ngram, *args, **kwargs):
    """
    :type collection_ngram: Collocation
    :type ngram: ArticleCollocation
    """
    return ngram.count, {}, {}


def abs_collection_count_score(collection_ngram, *args, **kwargs):
    """
    :type collection_ngram: Collocation
    """
    return collection_ngram.count, {}, {}


def abs_multi_count_score(collection_ngram, ngram, text, article_dict,
                           ngram_abs_count, *args, **kwargs):
    """
    :type collection_ngram: Collocation
    :type ngram: ArticleCollocation
    :type ngram_abs_count: int
    """
    return ngram_abs_count * collection_ngram.count, {}, {}


def C_value_score(collection_ngram, ngram, *args, **kwargs):
    """
    :type collection_ngram: Collocation
    :type ngram: ArticleCollocation
    """
    ngram_count = collection_ngram.count
    bigger_ngrams = collection_ngram.__class__.objects.filter(ngram__contains=ngram.ngram).values_list('count', flat=True)

    return ngram_count - sum(bigger_ngrams)/len(bigger_ngrams), {}, {}


def linked_score(collection_ngram, ngram, text, article_dict, ngram_abs_count, corr_dict1=None,
                 corr_dict2=None, score_func='weight_both_ngram4'):
    """
    :type collection_ngram: Collocation
    :type ngram: ArticleCollocation
    :type text: unicode
    """
    ngram = ngram.ngram
    nb = NgramBindings(ngram, text, corr_dict1=corr_dict1, corr_dict2=corr_dict2)
    if len(ngram.split()) == 2:
        score = getattr(nb, score_func)()
    else:
        smaller_ngrams = set(build_ngram_index(ngram).keys()).intersection(article_dict.keys())
        # select max split combination
        if smaller_ngrams:
            if len(smaller_ngrams) == 1:
                smaller_ngram = smaller_ngrams.pop()
                values = article_dict[smaller_ngram]
                score = values['score'] * ngram_abs_count / values['abs_count']
                # reduce score from the consumed ngram according its score
                article_dict[smaller_ngram]['score'] = values['score'] - score
                score = (score + getattr(nb, score_func)(split_ngram=smaller_ngram)) / 2
            else:
                score = 0
                smaller_ngrams = sorted(smaller_ngrams, key=lambda x: len(x.split()), reverse=True)
                for smaller_ngram in smaller_ngrams:
                    values = article_dict[smaller_ngram]
                    local_score = values['score'] * ngram_abs_count / values['abs_count']
                    article_dict[smaller_ngram]['score'] = values['score'] - local_score
                    score += local_score
        # no - full average
        else:
            score = getattr(nb, score_func)()
    return score, nb.ddict1, nb.ddict2


def linked_score1(collection_ngram, ngram, text, *args, **kwargs):
    """
    :type collection_ngram: Collocation
    :type ngram: ArticleCollocation
    :type text: unicode
    """
    ngram = ngram.ngram
    scores = []
    for i, word in enumerate(ngram.split()):

        regex = []
        for w in ngram.split():
            if w != word:
                regex.append(ur'{0}+'.format(NGRAM_REGEX))
            else:
                regex.append(word)
        regex = u' '.join(regex)
        distribution_dict = Counter(re.findall(regex, text, re.U))
        N = sum(distribution_dict.values())
        N_len = len(distribution_dict)
        score = distribution_dict[ngram]
        scores.append(score / N / N_len)
    return sum(scores) / len(scores), {}, {}


def linked_score2(collection_ngram, ngram, text, *args, **kwargs):
    """
    :type collection_ngram: Collocation
    :type ngram: ArticleCollocation
    :type text: unicode
    """
    ngram = ngram.ngram
    scores = []
    for i, word in enumerate(ngram.split()):

        regex = []
        for w in ngram.split():
            if w == word:
                regex.append(ur'{0}+'.format(NGRAM_REGEX))
            else:
                regex.append(w)
        regex = u' '.join(regex)
        distribution_dict = Counter(re.findall(regex, text, re.U))
        N = sum(distribution_dict.values())
        N_len = len(distribution_dict)
        score = distribution_dict[ngram]
        scores.append(score / N)
    print ngram
    print scores
    return sum(scores) / len(scores), {}, {}


class NgramBindings(object):

    def __init__(self, ngram, stemmed_text, corr_dict1=None, corr_dict2=None):
        self.values_dict = {}
        self.text = stemmed_text
        self.ngram = ngram
        self.corr_dict1 = corr_dict1 or {}
        self.corr_dict2 = corr_dict2 or {}
        self.ddict1 = {}
        self.ddict2 = {}

    def weight_both_ngram1(self, split_ngram=None):
        """AVERAGE BETWEEN TWO WEIGHTED SCORES"""
        ngram = self.ngram
        score = 0
        space_index = -1
        denominator = 0
        while True:
            space_index = ngram.find(' ', space_index + 1)
            if space_index == -1:
                break
            w1, w2 = ngram[:space_index], ngram[space_index + 1:]
            distribution_dict = Counter(re.findall(ur'({1}+) {0}'.format(w2, NGRAM_REGEX, re.U), self.text))
            N1 = sum(distribution_dict.values())
            N1_len = len(distribution_dict)
            score += distribution_dict[w1]
            distribution_dict = Counter(re.findall(ur'{0} ({1}+)'.format(w1, NGRAM_REGEX, re.U), self.text))
            N2 = sum(distribution_dict.values())
            N2_len = len(distribution_dict)
            score += distribution_dict[w2]
            score /= (N1 + N2)
            denominator += 1
        return score / denominator

    def weight_both_ngram2(self, split_ngram=None):
        """MODIFIER DETERMINER"""
        ngram = self.ngram
        score = 0
        space_index = -1
        denominator = 0
        while True:
            space_index = ngram.find(' ', space_index + 1)
            if space_index == -1:
                break
            w1, w2 = ngram[:space_index], ngram[space_index + 1:]
            distribution_dict = Counter(re.findall(ur'({1}+) {0}'.format(w2, NGRAM_REGEX), self.text, re.U))
            N2 = sum(distribution_dict.values())
            N2_len = len(distribution_dict)
            score1 = distribution_dict[w1]
            distribution_dict = Counter(re.findall(ur'{0} ({1}+)'.format(w1, NGRAM_REGEX), self.text, re.U))
            N1 = sum(distribution_dict.values())
            N1_len = len(distribution_dict)
            score2 = distribution_dict[w2]
            score = (score1 + score2) / (N1 + N2)
            # 4% increase in MAP
            #if N2_len / N1_len >= 5:
            #if N1_len / N2_len >= 5:
            #    score += 1
            score /= N2_len/N1_len
            denominator += 1
        return score / denominator

    def weight_both_ngram3(self, split_ngram=None):
        """MODIFIER DETERMINER2 TRIGRAMS OPTIMIZER"""
        ngram = self.ngram
        score = 0
        space_index = -1
        denominator = 0
        while True:
            space_index = ngram.find(' ', space_index + 1)
            if space_index == -1:
                break
            w1, w2 = ngram[:space_index], ngram[space_index + 1:]
            if split_ngram and w1 != split_ngram and not w2 != split_ngram:
                continue
            # subtract existing prefixes/suffixes from distribution dicts
            regex = u'(' + (ur'{0}+ '.format(NGRAM_REGEX) * len(w1.split()))[:-1] + u') ' + w2
            distribution_dict = Counter(re.findall(regex, self.text, re.U))
            N2 = sum(distribution_dict.values())
            N2_len = len(distribution_dict)
            score1 = distribution_dict[w1]

            regex = w1 + u' (' + (ur' {0}+'.format(NGRAM_REGEX) * len(w2.split()))[1:] + u')'
            distribution_dict = Counter(re.findall(regex, self.text, re.U))
            N1 = sum(distribution_dict.values())
            N1_len = len(distribution_dict)
            score2 = distribution_dict[w2]
            score += (score1 + score2) / (N1 + N2)
            # 4% increase in MAP
            score /= N2_len/N1_len
            denominator += 1
        return score / denominator

    def weight_both_ngram4(self, split_ngram=None):
        """MODIFIER DETERMINER + SUBTRACT EXISTING"""
        ngram = self.ngram
        score = 0
        space_index = -1
        denominator = 0
        while True:
            space_index = ngram.find(' ', space_index + 1)
            if space_index == -1:
                break
            w1, w2 = ngram[:space_index], ngram[space_index + 1:]
            if split_ngram and w1 != split_ngram and not w2 != split_ngram:
                continue
            # subtract existing prefixes/suffixes from distribution dicts
            regex = u'(' + (ur'{0}+ '.format(NGRAM_REGEX) * len(w1.split()))[:-1] + u') ' + w2
            self.ddict1 = distribution_dict = Counter(re.findall(regex, self.text, re.U))
            if w2 in self.corr_dict1:
                for w in self.corr_dict1[w2]:
                    if w != w1:
                        distribution_dict.pop(w, None)
            N2 = sum(distribution_dict.values())
            N2_len = len(distribution_dict)
            score1 = distribution_dict[w1]

            regex = w1 + u' (' + (ur' {0}+'.format(NGRAM_REGEX) * len(w2.split()))[1:] + u')'
            self.ddict2 = distribution_dict = Counter(re.findall(regex, self.text, re.U))
            if w1 in self.corr_dict2:
                for w in self.corr_dict2[w1]:
                    if w != w2:
                        distribution_dict.pop(w, None)
            N1 = sum(distribution_dict.values())
            N1_len = len(distribution_dict)
            score2 = distribution_dict[w2]
            score += (score1 + score2) / (N1 + N2)
            # 4% increase in MAP
            score /= N2_len/N1_len
            denominator += 1
        return score / denominator


def populate_article_dict(model, score_func, cutoff=1):
    """
    :type model: Model
    """
    article_dict = defaultdict(dict)
    for article in Article.objects.filter(cluster_id=model.CLUSTER_ID):
        text = article.stemmed_text
        # create correspondence dict
        corr_dict1 = defaultdict(set)
        corr_dict2 = defaultdict(set)
        all_ngrams = list(article.articlecollocation_set.values_list('ngram', flat=True))
        for ngram in all_ngrams:
            if len(ngram.split()) == 2:
                w1, w2 = ngram.split()
                corr_dict1[w2].add(w1)
                corr_dict2[w1].add(w2)
        for ngram in sorted(article.articlecollocation_set.all(),
                            key=lambda x: len(x.ngram.split())):
            part_count = 0
            for p_ngram in all_ngrams:
                if p_ngram != ngram.ngram and ngram.ngram in p_ngram:
                    part_count += 1
            is_rel = ngram.is_relevant
            if is_rel == -1:
                continue
            ngram_abs_count = text.count(ngram.ngram)
            if ngram_abs_count <= cutoff:
                continue
            collection_ngram = model.COLLECTION_MODEL.objects.get(ngram=ngram.ngram)
            score, ddict1, ddict2 = score_func(collection_ngram, ngram, text, article_dict[article],
                                               ngram_abs_count, corr_dict1, corr_dict2)
            nl_ngrams = [' '.join(n) for n in nltk.ngrams(ngram.ngram.split(), 2)]
            support_len = len(set(all_ngrams).intersection(nl_ngrams))
            article_dict[article][ngram.ngram] = {'abs_count': ngram_abs_count, 'score': score,
                                                  'is_rel': is_rel, 'count': ngram.count,
                                                  'ddict1': ddict1, 'ddict2': ddict2,
                                                  'ngram': collection_ngram,
                                                  'len': support_len,
                                                  'participation_count': part_count}

    return article_dict


def caclculate_MAP(article_dict):
    avg_prec_list = []

    for k, values_dict in article_dict.items():
        sorted_scores = OrderedDict(sorted(values_dict.iteritems(), key=lambda x: x[1]['score'],
                                           reverse=True))
        rel_ngram_num = len([x for x in sorted_scores.values() if x['is_rel']])
        if rel_ngram_num == 0:
            continue
        correct_count = 0
        local_precision = 0
        i = 0
        for ngram, values in sorted_scores.items():
            i += 1
            if values['is_rel']:
                correct_count += 1
                local_precision += correct_count / i
        avg_prec_list.append(local_precision/rel_ngram_num)
    return sum(avg_prec_list) / len(avg_prec_list)
