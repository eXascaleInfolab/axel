from __future__ import division
import os
import re
import pickle
from collections import OrderedDict, defaultdict
from sklearn import cross_validation, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from axel.stats.scores import compress_pos_tag
from optparse import make_option
import numpy as np
import networkx as nx

from django.core.management.base import BaseCommand, CommandError

from axel.articles.models import Article
from axel.stats.scores.binding_scores import populate_article_dict, caclculate_MAP
from axel.stats.scores import binding_scores


RULES_DICT = OrderedDict([(u'NUMBER', re.compile('CD')), (u'ADVERB_CONTAINS', re.compile('RB')),
                          (u'STOP_WORD', re.compile(r'(NONE|DT|CC|MD|RP)')),
                          (u'AJD_FORM', re.compile(r'JJR|JJS')),
                          (u'PREP_START', re.compile(r'(^IN|IN$)')),
                          (u'4NGRAM', re.compile(r'([A-Z]{2}.? ?){4}')),
                          (u'5NGRAM', re.compile(r'([A-Z]{2}.? ?){5}')),
                          (u'PLURAL_START', re.compile(r'^NNS')),
                          (u'VERB_STARTS', re.compile(r'^VB')),
                          (u'NN_STARTS', re.compile(r'^NN')),
                          (u'JJ_STARTS', re.compile(r'^JJ'))]) # JJ reduces classification accuracy by 5%

POS_TAG_DIST = {'JJ_STARTS': 1.23495702006, 'VERB_STARTS': 0.322884012539, '4NGRAM': 0.196319018405,
                'NUMBER': 0.0283505154639, 'ADVERB_CONTAINS': 0.04, 'NN_STARTS': 3.74587458746,
                'STOP_WORD': 0.0, 'PLURAL_START': 0.226890756303, 'AJD_FORM': 0.0675675675676,
                'PREP_START': 0.209302325581}


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--cluster', '-c',
                    action='store',
                    dest='cluster',
                    help='cluster id for article type'),
        make_option('--classify',
                    action='store_true',
                    dest='classify',
                    default=False,
                    help='whether to classify collection instead of MAP calculation'),
        make_option('--cvnum', '-n',
                    action='store',
                    dest='cv_num',
                    type='int',
                    default=10,
                    help='number of cross validation folds, defaults to 10'),
    )
    args = '<score1> <score2> ...'
    help = 'Train SVM classified with a set of features'

    def _total_valid(self, article_dict):
        total_valid = 0
        for values in article_dict.itervalues():
            for value in values.itervalues():
                total_valid += int(value['is_rel'])
        return total_valid

    def handle(self, *args, **options):

        self.cluster_id = cluster_id = options['cluster']
        if not cluster_id:
            raise CommandError("need to specify cluster id")
        cv_num = options['cv_num']
        self.Model = Model = Article.objects.filter(cluster_id=cluster_id)[0].CollocationModel
        for score_name in args:
            print 'Building initial binding scores for {0}...'.format(score_name)
            if os.path.exists('article_dict.pcl'):
                print 'File found, loading...'
                article_dict = pickle.load(open('article_dict.pcl'))
            else:
                article_dict = dict(populate_article_dict(Model.objects.values_list('ngram', flat=True),
                                                 getattr(binding_scores, score_name), cutoff=0))
                pickle.dump(article_dict, open('article_dict.pcl', 'wb'))
            # Calculate total valid for recall
            total_valid = self._total_valid(article_dict)

            if options['classify']:
                scored_ngrams = []
                print 'Reformatting the results...'
                for article, values in article_dict.iteritems():
                    for scores in values.itervalues():
                        scored_ngrams.append((article, scores))

                print 'Fitting classifier...'
                fit_ml_algo(scored_ngrams, cv_num)
            else:
                map_results = []
                for i in range(50):
                    print 'Cutoff {0}'.format(i)
                    article_dict = dict([(article, dict((ngram, value) for ngram, value in values.iteritems() if value['abs_count'] > i))
                                         for article, values in article_dict.iteritems()])
                    # adjust score:
                    for article, values in article_dict.iteritems():
                        for ngram, value in values.iteritems():
                            source = 1/100
                            if 'dblp' in value['ngram'].source:
                                source += 1
                            if 'dbpedia' in value['ngram'].source:
                                source += 1
                            value['score'] = value['abs_count'] * POS_TAG_DIST[compress_pos_tag(value['ngram'].pos_tag, RULES_DICT)]*(100*source)

                    map_score = caclculate_MAP(article_dict)
                    map_results.append((i, map_score))
                print str(map_results).replace('(', '[').replace(')', ']')


def fit_ml_algo(scored_ngrams, cv_num):
    """
    :param scored_ngrams: list of tuple of type (ngram, score) after initial scoring
    """
    # 1. Calculate scores with float numbers for ngram bindings, as a dict
    collection = []
    collection_labels = []
    pos_tag_dict = {}
    pos_tag_i = 0
    max_pos_tag = 10
    component_size_dict = {}
    # 2. Iterate through all ngrams, add scores - POS tag (to number), DBLP, DBPEDIA, IS_REL
    for article, score_dict in scored_ngrams:
        temp_dict = defaultdict(lambda: 0)
        if article.id not in component_size_dict:
            dbpedia_graph = article.dbpedia_graph
            for component in nx.connected_components(dbpedia_graph):
                comp_len = len(component)
                for node in component:
                    if 'Category' not in node:
                        temp_dict[node] = comp_len
            component_size_dict[article.id] = temp_dict
        ngram = score_dict['ngram']
        pos_tag = str(compress_pos_tag(ngram.pos_tag, RULES_DICT))
        if pos_tag not in pos_tag_dict:
            pos_tag_dict[pos_tag] = pos_tag_i
            pos_tag_i += 1

        wiki_edges_count = len(article.wikilinks_graph.edges([ngram.ngram]))

        feature = [ngram.count, score_dict['abs_count'], wiki_edges_count,
                   component_size_dict[article.id][ngram.ngram], 'dblp' in ngram.source]
        # extend with part of speech
        extended_feature = [1 if i == pos_tag_dict[pos_tag] else 0 for i in range(max_pos_tag)]
        feature.extend(extended_feature)

        collection.append(feature)
        collection_labels.append(score_dict['is_rel'])
    #clf = svm.SVC(kernel='linear')
    clf = DecisionTreeClassifier(min_samples_leaf=100, max_depth=5)
    #for tag, values in pos_tag_counts.iteritems():
    #    print tag, values[1]/values[0]
    # clf.fit(collection, collection_labels)
    # import StringIO, pydot
    # from sklearn import tree
    # dot_data = StringIO.StringIO()
    # feature_names = ['col_count', 'abs_count', 'wikilinks', 'comp_size', 'dblp']
    # feature_names.extend([str(i) for i in range(max_pos_tag)])
    # tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("decision.pdf")
    #
    # for i, vector in enumerate(collection):
    #     value = clf.predict(vector)[0]
    #     if value != collection_labels[i]:
    #         print vector, value, collection_labels[i]

    # K-fold cross-validation
    print 'Performing cross validation'
    scores = cross_validation.cross_val_score(clf, collection, np.array(collection_labels),
                                              cv=cv_num)

    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() / 2))
