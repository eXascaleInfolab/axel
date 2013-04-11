import re
from collections import OrderedDict
from sklearn import svm, cross_validation
from axel.stats.scores import compress_pos_tag
from optparse import make_option
import numpy as np

from django.core.management.base import BaseCommand, CommandError

from axel.articles.models import Article
from axel.stats.scores.binding_scores import populate_article_dict, caclculate_MAP
from axel.stats.scores import binding_scores


RULES_DICT = OrderedDict([(u'NUMBER', re.compile('CD')), (u'ADVERB_CONTAINS', re.compile('RB')),
                          (u'STOP_WORD', re.compile(r'(NONE|DT|CC|MD|RP)')),
                          (u'AJD_FORM', re.compile(r'JJR|JJS')),
                          (u'SMTH_BEFORE_ADJ', re.compile(r'\sJJ')),
                          (u'4NGRAM', re.compile(r'([A-Z]{2}.? ?){4}')),
                          (u'5NGRAM', re.compile(r'([A-Z]{2}.? ?){5}')),
                          (u'PLURAL_START', re.compile(r'^NNS')),
                          (u'PROPER', re.compile(r'(NNP\s?)+')),
                          (u'PREP_END', re.compile(r'IN $'))])


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

    def handle(self, *args, **options):

        self.cluster_id = cluster_id = options['cluster']
        if not cluster_id:
            raise CommandError("need to specify cluster id")
        cv_num = options['cv_num']
        self.Model = Model = Article.objects.filter(cluster_id=cluster_id)[0].CollocationModel
        for score_name in args:
            print 'Building initial binding scores for {0}...'.format(score_name)
            article_dict = populate_article_dict(Model.objects.values_list('ngram', flat=True),
                                                 getattr(binding_scores, score_name), cutoff=0)
            if options['classify']:
                scored_ngrams = []
                print 'Reformatting the results...'
                for values in article_dict.itervalues():
                    for ngram, scores in values.iteritems():
                        ngram_count, score, is_rel = scores
                        ngram_obj = Model.objects.get(ngram=ngram)
                        scored_ngrams.append((ngram_obj, score, ngram_count, is_rel))

                print 'Fitting classifier...'
                fit_ml_algo(scored_ngrams, cv_num)
            else:
                map_results = []
                for i in range(50):
                    print 'Cutoff {0}'.format(i)
                    article_dict = dict([(article, dict((ngram, value) for ngram, value in values.iteritems() if value['abs_count'] > i))
                                         for article, values in article_dict.iteritems()])
                    map_score = caclculate_MAP(article_dict)
                    # TODO: calculate recall
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
    # 2. Iterate through all ngrams, add scores - POS tag (to number), DBLP, DBPEDIA, IS_REL
    for ngram, initial_score, count, is_rel in scored_ngrams:
        pos_tag = str(compress_pos_tag(ngram.pos_tag, RULES_DICT))
        if pos_tag not in pos_tag_dict:
            pos_tag_dict[pos_tag] = pos_tag_i
            pos_tag_i += 1
        collection.append((initial_score, 'dblp' in ngram.source, 'dbpedia' in ngram.source,
                           pos_tag_dict[pos_tag], count))
        collection_labels.append(is_rel)
    clf = svm.SVC(kernel='linear', probability=True)
    #clf.fit(collection, collection_labels)

    # K-fold cross-validation
    scores = cross_validation.cross_val_score(clf, collection, np.array(collection_labels),
                                              cv=cv_num)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
