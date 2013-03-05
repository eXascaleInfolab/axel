# coding=utf-8
"""Produces graph of different ordering of concept inside article comparison"""
from __future__ import division

from collections import defaultdict
import json
import nltk

from test_collection.models import TaggedCollection
from axel.articles.models import Article
from axel.articles.utils.nlp import _update_ngram_counts, _generate_possible_ngrams, _PUNKT_RE,\
    _DIGIT_RE, _STOPWORDS
from axel.libs.utils import print_progress

bigram_measures = nltk.collocations.BigramAssocMeasures

MEASURES = (('Student T', bigram_measures.student_t), ('Pearson χ', bigram_measures.chi_sq),
            ('MI likelihood', bigram_measures.mi_like), ('Pointwise MI', bigram_measures.pmi),
            ('Likelihood ratio', bigram_measures.likelihood_ratio),
            ('Jaccard', bigram_measures.jaccard), ('Dice', bigram_measures.dice),
            ('Poisson-Stirling', bigram_measures.poisson_stirling))


class NgramMeasureScoring:

    @staticmethod
    def collocations(index, measures):
        """
        Extract collocations from n-gram index
        :type index: dict
        :rtype list
        """

        def filter_punkt(word):
            return _PUNKT_RE.match(word)

        def filter_len(word):
            return len(word) < 3 and not word.isupper()

        # do filtration by frequency > 2
        bigram_index = dict([(tuple(k.split()), v) for k, v in index.iteritems()
                             if len(k.split()) == 2 and v > 2])

        # Get abstract finder because we already have index
        finder = nltk.collocations.AbstractCollocationFinder(None, bigram_index)
        # remove collocation from 2 equal words
        finder.apply_ngram_filter(lambda x, y: x == y)
        # remove weird collocations
        finder.apply_ngram_filter(lambda x, y: _DIGIT_RE.match(x) and _DIGIT_RE.match(y))
        # remove punctuation, len and stopwords
        finder.apply_word_filter(filter_punkt)
        finder.apply_word_filter(filter_len)
        finder.apply_word_filter(lambda w: w in _STOPWORDS)

        # build word distribution
        from nltk.probability import FreqDist
        word_fd = FreqDist()
        for ngram, count in index.iteritems():
            for word in ngram.split():
                word_fd.inc(word, count)
        finder_big = nltk.collocations.BigramCollocationFinder(word_fd, finder.ngram_fd)

        filtered_collocs = _update_ngram_counts(_generate_possible_ngrams(finder.ngram_fd, index),
                                                index).items()
        filtered_collocs.sort(key=lambda col: col[1], reverse=True)
        filtered_collocs = zip(*filtered_collocs)[0]  # keep only ordering, we don't need score
        yield 'raw', filtered_collocs

        # build bigram correspondence dict
        corr_dict = defaultdict(list)
        for ngram in filtered_collocs:
            for bigram in nltk.ngrams(ngram.split(), 2):
                corr_dict[bigram].append(ngram)

        for measure_name, measure_func in measures:
            bigrams = finder_big.score_ngrams(measure_func)
            scored_ngrams = defaultdict(lambda: 0)
            for bigram, score in bigrams:
                for ngram in corr_dict[bigram]:
                    scored_ngrams[ngram] += score
            scored_ngrams = scored_ngrams.items()
            scored_ngrams.sort(key=lambda col: col[1], reverse=True)
            yield measure_name, zip(*scored_ngrams)[0]

    @classmethod
    def get_scores(cls, model):
        """
        :param model: Collocation
        :returns: accuracy scores for each method
        :rtype: defaultdict
        """

        relevant_names = set(TaggedCollection.objects.filter(is_relevant=True).values_list(
            model.__name__.lower() + '__ngram', flat=True))
        irrelevant_names = set(TaggedCollection.objects.filter(is_relevant=False).values_list(
            model.__name__.lower() + '__ngram', flat=True))

        unjudged = defaultdict(lambda: 0)
        orderings = defaultdict(lambda: {'relevant': defaultdict(lambda: 0),
                                         'irrelevant': defaultdict(lambda: 0)})

        print 'Starting article processing...'
        for article in print_progress(Article.objects.filter(cluster_id=model.CLUSTER_ID)):
            index = json.loads(article.index)
            for order_name, ordering in cls.collocations(index, MEASURES):
                for i, ngram in enumerate(ordering):
                    if ngram in relevant_names:
                        orderings[order_name]['relevant'][i] += 1
                    elif ngram in irrelevant_names:
                        orderings[order_name]['irrelevant'][i] += 1
                    else:
                        # not present
                        unjudged[i] += 1
        print 'End article processing...'
        print 'Starting result formatting...'

        graph_results = defaultdict(list)
        for order_name, results in orderings.iteritems():
            total_relevant = 0
            total_irrelevant = 0
            for rel_count, irrel_count in zip(results['relevant'].items(),
                                              results['irrelevant'].items()):
                total_relevant += rel_count[1]
                total_irrelevant += irrel_count[1]
                graph_results[order_name].append((rel_count[0],
                    round(total_relevant / (total_irrelevant + total_relevant), 3)))

        return graph_results

