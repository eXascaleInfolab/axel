from __future__ import division
import os
import re
import pickle
from collections import defaultdict
from sklearn import cross_validation, svm
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier
from axel.stats.scores import compress_pos_tag
from optparse import make_option
import numpy as np
import networkx as nx

from django.core.management.base import BaseCommand, CommandError

from axel.articles.models import Article
from axel.stats.scores.binding_scores import populate_article_dict
from axel.stats.scores import binding_scores


RULES_DICT_START = [(u'STOP_WORD', re.compile(r'(NONE|DT|CC|MD|RP|JJR|JJS|\:)')),
                    (u'NUMBER_STARTS', re.compile('^CD')),
                    (u'ADVERB_STARTS', re.compile('^RB')),
                    (u'PREP_START', re.compile(r'(^IN)')),
                    (u'NNS_START', re.compile(r'^NNS')),
                    (u'VB_STARTS', re.compile(r'^VB')),
                    (u'NN_STARTS', re.compile(r'^NN')),
                    (u'JJ_STARTS', re.compile(r'^JJ'))]
RULES_DICT_END = [(u'STOP_WORD', re.compile(r'(NONE|DT|CC|MD|RP|JJR|JJS|\:)')),
                  (u'NUMBER_ENDS', re.compile('CD$')),
                  (u'ADVERB_ENDS', re.compile('RB.?$')),
                  (u'PREP_ENDS', re.compile(r'(IN$)')),
                  (u'NNS_ENDS', re.compile(r'NNS$')),
                  (u'VB_ENDS', re.compile(r'VB.?$')),
                  (u'NN_ENDS', re.compile(r'NN(P|PS)?$')),
                  (u'JJ_ENDS', re.compile(r'JJ.?$'))]


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

            scored_ngrams = []
            print 'Reformatting the results...'
            for article, values in article_dict.iteritems():
                for scores in values.itervalues():
                    scored_ngrams.append((article, scores))

            print 'Fitting classifier...'
            fit_ml_algo(scored_ngrams, cv_num, self.Model)


def fit_ml_algo(scored_ngrams, cv_num, Model):
    """
    :param scored_ngrams: list of tuple of type (ngram, score) after initial scoring
    """
    # 1. Calculate scores with float numbers for ngram bindings, as a dict
    collection = []
    collection_labels = []
    component_size_dict = {}

    # Calculate max pos tag count and build pos_tag_list
    start_pos_tag_list = []
    end_pos_tag_list = []
    pos_tag_list = []
    for ngram in Model.objects.all():
        pos_tag_start = str(compress_pos_tag(ngram.pos_tag, RULES_DICT_START))
        pos_tag_end = str(compress_pos_tag(ngram.pos_tag, RULES_DICT_END))
        pos_tag = str(ngram.pos_tag)
        if pos_tag_start not in start_pos_tag_list:
            start_pos_tag_list.append(pos_tag_start)
        if pos_tag_end not in end_pos_tag_list:
            end_pos_tag_list.append(pos_tag_end)
        if pos_tag not in pos_tag_list:
            pos_tag_list.append(pos_tag)
    max_pos_tag_start = len(start_pos_tag_list)
    max_pos_tag_end = len(end_pos_tag_list)
    max_pos_tag = len(pos_tag_list)

    # 2. Iterate through all ngrams, add scores - POS tag (to number), DBLP, DBPEDIA, IS_REL
    for article, score_dict in scored_ngrams:
        temp_dict = defaultdict(lambda: 0)
        if article.id not in component_size_dict:
            dbpedia_graph = article.dbpedia_graph(redirects=True)
            for component in nx.connected_components(dbpedia_graph):
                comp_len = len([node for node in component if 'Category' not in node])
                for node in component:
                    temp_dict[node] = comp_len
            component_size_dict[article.id] = temp_dict
        ngram = score_dict['ngram']

        # POS TAG enumeration
        pos_tag_start = str(compress_pos_tag(ngram.pos_tag, RULES_DICT_START))
        pos_tag_end = str(compress_pos_tag(ngram.pos_tag, RULES_DICT_END))
        pos_tag = str(ngram.pos_tag)

        wiki_edges_count = len(article.wikilinks_graph.edges([ngram.ngram]))

        feature = [
                   ngram.ngram.isupper(),
                   'dblp' in ngram.source,
                   component_size_dict[article.id][ngram.ngram],
                   wiki_edges_count,
                   score_dict['participation_count'],
                   ngram._is_wiki,
                   ngram.is_ontological,
                   'dbpedia' in ngram.source,
                   'wiki_redirect' in ngram.source,
                   bool({'.', ',', ':', ';'}.intersection(ngram.pos_tag_prev.keys())),
                   bool({'.', ',', ':', ';'}.intersection(ngram.pos_tag_after.keys())),
                   len(ngram.ngram.split())
        ]

        # extend with compressed part of speech
        extended_feature = [1 if i == start_pos_tag_list.index(pos_tag_start) else 0 for i in range(max_pos_tag_start)]
        feature.extend(extended_feature)
        extended_feature = [1 if i == end_pos_tag_list.index(pos_tag_end) else 0 for i in range(max_pos_tag_end)]
        feature.extend(extended_feature)

        # Normal part of speech
        # extended_feature = [1 if i == pos_tag_list.index(pos_tag) else 0 for i in range(max_pos_tag)]
        # feature.extend(extended_feature)

        collection.append(feature)
        collection_labels.append(score_dict['is_rel'])
    #clf = svm.SVC(kernel='linear')

    # from sklearn.feature_selection import RFECV
    # from sklearn.metrics import zero_one_loss
    # svc = svm.SVC(kernel="linear")
    # rfecv = RFECV(estimator=svc, step=1, cv=2, loss_func=zero_one_loss)
    # rfecv.fit(collection, collection_labels)
    # print("Optimal number of features : %d" % rfecv.n_features_)
    # print rfecv.ranking_
    # import pylab as pl
    # pl.figure()
    # pl.xlabel("Number of features selected")
    # pl.ylabel("Cross validation score (N of misclassifications)")
    # pl.plot(range(1, len(rfecv.cv_scores_) + 1), rfecv.cv_scores_)
    # pl.show()

    feature_names = [
        'is_upper',
        'dblp',
        'comp_size',
        'wikilinks',
        'part_count',
        'is_wiki',
        'ScienceWISE',
        'dbpedia',
        'is_redirect',
        'pos_tag_prev',
        'pos_tag_after',
        'word_len'
    ]
    feature_names.extend(start_pos_tag_list)
    feature_names.extend(end_pos_tag_list)
    # feature_names.extend(pos_tag_list)

    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier(random_state=0, compute_importances=True, n_estimators=20)
    new_collection = clf.fit(collection, collection_labels).transform(collection)
    print sorted(zip(list(clf.feature_importances_), feature_names), key=lambda x: x[0],
                 reverse=True)[:new_collection.shape[1]]
    print new_collection.shape
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)
    #for tag, values in pos_tag_counts.iteritems():
    #    print tag, values[1]/values[0]
    clf.fit(collection, collection_labels)
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    #feature_names = ['dblp', 'comp_size', 'NN_STARTS', 'NN_STARTS', 'test', 'test', 'test']
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("decision.pdf")
    #
    # for i, vector in enumerate(collection):
    #     value = clf.predict(vector)[0]
    #     if value != collection_labels[i] and value:
    #         print scored_ngrams[i][1]['ngram'], vector, value, collection_labels[i]

    # K-fold cross-validation
    print 'Performing cross validation'
    scores = cross_validation.cross_val_score(clf, new_collection, np.array(collection_labels),
                                              cv=cv_num, score_func=precision_score)
    print("Precision: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() / 2))
    scores = cross_validation.cross_val_score(clf, new_collection, np.array(collection_labels),
                                              cv=cv_num, score_func=recall_score)
    print("Recall: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() / 2))
    scores = cross_validation.cross_val_score(clf, new_collection, np.array(collection_labels),
                                              cv=cv_num, score_func=f1_score)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() / 2))

