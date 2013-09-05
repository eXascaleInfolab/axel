"""Match extracted collocation with DBPedia entities"""
from __future__ import division
from collections import Counter, defaultdict
from optparse import make_option
from termcolor import colored
import nltk

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from axel.articles.models import Article, CLUSTERS_DICT
from axel.libs import nlp
from axel.libs.utils import print_progress


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--cluster', '-c',
                    action='store',
                    dest='cluster',
                    help='cluster id for article type'),
        make_option('--redirects',
                    action='store_true',
                    dest='redirects',
                    default=False,
                    help='[DBPedia only] Calculate with/without redirects'),
    )

    args = '<method1> <method2> ...'
    help = 'Calculated P/R metrics between extracted collocations and chosen method'

    def handle(self, *args, **options):

        self.cluster_id = cluster_id = options['cluster']
        if not cluster_id:
            raise CommandError("need to specify cluster id")
        self.Model = CLUSTERS_DICT[cluster_id]
        self.redirects = options['redirects']

        self.article_rel_dict = defaultdict(lambda: {0: set(), 1: set()})
        for key, is_rel in self.Model.judged_data.iteritems():
            ngram, article_id = key.split(',')
            is_rel = int(is_rel)
            self.article_rel_dict[article_id][is_rel].add(ngram)

        for arg in args:
            eval_method = getattr(self, '_' + arg + '_calculation')
            eval_method()

    def _wikilinks_calculation(self):
        print 'Generating wikilinks Histogram'
        relation_distibution = {'valid': 0, 'invalid': 0, 'multi': 0}

        dbpedia_ngrams = set()
        for colloc in self.Model.objects.all():
            if 'dbpedia' in colloc.source:
                dbpedia_ngrams.add(colloc.ngram)

        for article in Article.objects.filter(cluster_id=self.cluster_id):
            print article

            correct_objects = self.article_rel_dict[unicode(article)][1]
            incorrect_objects = self.article_rel_dict[unicode(article)][0]
            article_ngrams = list(self.Model.objects.filter(article=article).values_list('ngram', flat=True))
            links_graph = article.wikilinks_graph

            for i, ngram1 in enumerate(article_ngrams):
                try:
                    ngram1_links = zip(*links_graph.edges(nbunch=[ngram1]))[1]
                except IndexError:
                    ngram1_links = set()
                for j in range(i+1, len(article_ngrams)):
                    ngram2 = article_ngrams[j]
                    try:
                        ngram2_links = zip(*links_graph.edges(nbunch=[ngram2]))[1]
                    except IndexError:
                        ngram2_links = set()

                    if ngram1 in correct_objects and ngram2 in correct_objects:
                        attr = 'valid'
                    elif ngram1 in incorrect_objects and ngram2 in incorrect_objects:
                        attr = 'invalid'
                    else:
                        attr = 'multi'
                    if ngram1 in ngram2_links or ngram2 in ngram1_links:
                        #if attr == 'multi' or attr == 'invalid':
                        #    print ngram1, ngram2
                        relation_distibution[attr] += 1
        print relation_distibution

    def _dbpedia_cc_size_calculation(self):
        print 'Generating DBPedia Connected Component size Histogram'
        import networkx as nx

        cc_size_distibution = {'valid': defaultdict(lambda: 0), 'invalid': defaultdict(lambda: 0)}
        percentage_distribution = defaultdict(list)

        for article in Article.objects.filter(cluster_id=self.cluster_id):
            print article
            graph = article.dbpedia_graph()
            results = []

            correct_objects = self.article_rel_dict[unicode(article)][1]
            incorrect_objects = self.article_rel_dict[unicode(article)][0]
            for component in nx.connected_components(graph):
                component = [node for node in component if 'Category' not in node]
                results.append(component)
                conn_valid = len([node for node in component if node in correct_objects])
                conn_invalid = len([node for node in component if node in incorrect_objects])
                if (conn_valid + conn_invalid) == 0:
                    print component
                    continue
                percentage_distribution[len(component)].append(conn_valid/(conn_valid+conn_invalid))
                if conn_valid > conn_invalid:
                    cc_size_distibution['valid'][len(component)] += 1
                else:
                    cc_size_distibution['invalid'][len(component)] += 1
        print 'VALID_PER', str([(item, sum(per_list)/len(per_list)) for item, per_list in percentage_distribution.items()]).replace('(','[').replace(')', ']')
        print 'INVALID_PER', str([(item, 1-sum(per_list)/len(per_list)) for item, per_list in percentage_distribution.items()]).replace('(','[').replace(')', ']')
        print 'VALID', str(cc_size_distibution['valid'].items()).replace('(','[').replace(')', ']')
        print 'INVALID', str(cc_size_distibution['invalid'].items()).replace('(','[').replace(')', ']')

    def _dbpedia_calculation(self):
        """
        Calculates P/R for dbpedia concepts, it is possible to exclude different concepts based
        on their biggest connected component for example. Then it will print which correct concepts
        were incorrectly excluded.
        """
        print 'Calculating Precision/Recall using dbpedia graph method'
        precision = []
        recall = []

        dbpedia_ngrams = set()
        for colloc in self.Model.COLLECTION_MODEL.objects.all():
            if 'dbpedia' in colloc.source:
                dbpedia_ngrams.add(colloc.ngram)

        top_false_counter = Counter()
        for article in Article.objects.filter(cluster_id=self.cluster_id):
            results = set(article.dbpedia_graph(redirects=self.redirects).nodes())

            article_ngrams = self.Model.objects.filter(article=article).values_list('ngram', flat=True)
            correct_objects = self.article_rel_dict[unicode(article)][1]
            incorrect_objects = self.article_rel_dict[unicode(article)][0]
            all_dbpedia_ngrams = [ngram for ngram in article_ngrams if ngram in dbpedia_ngrams]

            true_pos = [x for x in results if x in correct_objects]
            good_removed = [x for x in all_dbpedia_ngrams if x in correct_objects and x not in true_pos]
            false_pos = [x for x in results if x in incorrect_objects]
            top_false_counter.update(false_pos)
            local_precision = len(true_pos) / (len(true_pos) + len(false_pos))
            local_recall = len(true_pos) / len([x for x in article.articlecollocation_set
            .values_list('ngram', flat=True) if x in correct_objects])

            precision.append(local_precision)
            recall.append(local_recall)
            print article.id
            print colored(true_pos, 'green')
            print colored(good_removed, 'yellow')
            print colored(false_pos, 'red')
            print

        print precision
        print recall
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        print 'Length:', len(dbpedia_ngrams)
        print 'Precision: ', precision
        print 'Recall', recall
        print 'F1 measure', 2 * (precision * recall) / (precision + recall)
        print colored(str(top_false_counter), 'red')

    def _dblp_calculation(self):
        print 'Calculating Precision/Recall using DBLP match'
        precision = []
        recall = []
        dblp_ngrams = set()
        for colloc in self.Model.COLLECTION_MODEL.objects.all():
            if 'dblp' in colloc.source:
                dblp_ngrams.add(colloc.ngram)

        for article in Article.objects.filter(cluster_id=self.cluster_id):
            article_ngrams = self.Model.objects.filter(article=article).values_list('ngram', flat=True)
            results = [ngram for ngram in article_ngrams if ngram in dblp_ngrams]

            correct_objects = self.article_rel_dict[unicode(article)][1]
            incorrect_objects = self.article_rel_dict[unicode(article)][0]

            true_pos = [x for x in results if x in correct_objects]
            false_pos = [x for x in results if x in incorrect_objects]
            print article.id
            print colored(true_pos, 'green')
            print colored(false_pos, 'red')
            print
            local_precision = len(true_pos) / (len(true_pos) + len(false_pos))
            local_recall = len(true_pos) / len([x for x in article_ngrams if x in correct_objects])

            precision.append(local_precision)
            recall.append(local_recall)

        print precision
        print recall
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        print 'Length:', len(dblp_ngrams)
        print 'Precision: ', precision
        print 'Recall', recall
        print 'F1 measure', 2 * (precision * recall) / (precision + recall)

    def _maxent_calculation(self):
        print 'Calculating Precision/Recall using MaxEntropy NE recognition'
        precision = []
        recall = []

        extra_source = open(settings.ABS_PATH('maxent_' + self.Model.__name__ + '.csv')).read().split('\n')
        results_dict = defaultdict(lambda: {'true_pos': set(), 'false_pos': set()})
        for line in extra_source:
            line = line.split(',')
            if line[1] == '0':
                results_dict[line[3]]['false_pos'].add(line[0])
            else:
                results_dict[line[3]]['true_pos'].add(line[0])

        for article in Article.objects.filter(cluster_id=self.cluster_id):
            false_negs = self.article_rel_dict[unicode(article)][1]
            pdf_id = str(article.pdf)[12:-4]

            true_pos = results_dict[pdf_id]['true_pos']
            false_pos = results_dict[pdf_id]['false_pos']
            results_dict[pdf_id]['false_negs'] = false_negs.difference(true_pos)
            local_precision = len(true_pos) / (len(true_pos) + len(false_pos))
            local_recall = len(true_pos) / (len(true_pos) + len(false_negs))
            precision.append(local_precision)
            recall.append(local_recall)

        # We already have judgments, no need to extract
        # for article in Article.objects.filter(cluster_id=self.cluster_id):
        #     #print article
        #     text = article.text
        #     article_ngrams = self.Model.objects.filter(article=article).values_list('ngram', 'tags__is_relevant')
        #     correct_objects = [ngram for ngram, rel in article_ngrams if rel]
        #     incorrect_objects = [ngram for ngram, rel in article_ngrams if rel is False]
        #
        #     sentences = [nltk.pos_tag(nltk.regexp_tokenize(sent, nlp.Stemmer.TOKENIZE_REGEXP)) for sent in nltk.sent_tokenize(text)]
        #     results = nltk.batch_ne_chunk(sentences)
        #     """:type: nltk.tree.Tree"""
        #
        #     ne_set = set()
        #     for result in results:
        #         for tree in result.subtrees():
        #             if tree.node != 'S' and len(tree) > 1:
        #                 ne_set.add(nlp.Stemmer.stem_wordnet(' '.join(zip(*tree)[0]).lower()))
        #     true_pos = [x for x in ne_set if x in correct_objects]
        #     false_pos = [x for x in ne_set if x in incorrect_objects]
        #
        #     if len(true_pos) + len(false_pos) == 0:
        #         continue
        #
        #     local_precision = len(true_pos) / (len(true_pos) + len(false_pos))
        #     local_recall = len(true_pos) / len([x for x, _ in article_ngrams if x in correct_objects])
        #
        #     precision.append(local_precision)
        #     recall.append(local_recall)

        #     precision.append(local_precision)
        #     recall.append(local_recall)

        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        print 'Precision: ', precision
        print 'Recall', recall
        print 'F1 measure', 2 * (precision * recall) / (precision + recall)

    def _punct_calculation(self):
        print 'Calculating Contigency tables for after/before punctuation'
        pos_tag_prev = [0, 0, 0, 0]
        pos_tag_after = [0, 0, 0, 0]

        for article in print_progress(Article.objects.filter(cluster_id=self.cluster_id)):
            for ngram in self.Model.objects.filter(article=article):
                if ngram.ngram in self.article_rel_dict[unicode(article)][1]:
                    if {'.', ',', ':', ';'}.intersection(zip(*ngram.pos_tag_prev)[0]):
                        pos_tag_prev[0] += 1
                    else:
                        pos_tag_prev[2] += 1

                    if {'.', ',', ':', ';'}.intersection(zip(*ngram.pos_tag_after)[0]):
                        pos_tag_after[0] += 1
                    else:
                        pos_tag_after[2] += 1
                elif ngram.ngram in self.article_rel_dict[unicode(article)][0]:
                    if {'.', ',', ':', ';'}.intersection(zip(*ngram.pos_tag_prev)[0]):
                        pos_tag_prev[1] += 1
                    else:
                        pos_tag_prev[3] += 1

                    if {'.', ',', ':', ';'}.intersection(zip(*ngram.pos_tag_after)[0]):
                        pos_tag_after[1] += 1
                    else:
                        pos_tag_after[3] += 1

        print 'Contigency table BEFORE:'
        print '       | Valid | Invalid | Total |'
        print '+punct | {0:>5} | {1:>7} | {2:>5} |'.format(pos_tag_prev[0], pos_tag_prev[1],
                                                           pos_tag_prev[0] + pos_tag_prev[1])
        print '-punct | {0:>5} | {1:>7} | {2:>5} |'.format(pos_tag_prev[2], pos_tag_prev[3],
                                                           pos_tag_prev[2] + pos_tag_prev[3])
        print 'Totals | {0:>5} | {1:>7} | {2:>5} |'.format(pos_tag_prev[0] + pos_tag_prev[2],
                                                           pos_tag_prev[1] + pos_tag_prev[3],
                                                           sum(pos_tag_prev))
        print
        print 'Contigency table AFTER:'
        print '       | Valid | Invalid | Total |'
        print '+punct | {0:>5} | {1:>7} | {2:>5} |'.format(pos_tag_after[0], pos_tag_after[1],
                                                           pos_tag_after[0] + pos_tag_after[1])
        print '-punct | {0:>5} | {1:>7} | {2:>5} |'.format(pos_tag_after[2], pos_tag_after[3],
                                                           pos_tag_after[2] + pos_tag_after[3])
        print 'Totals | {0:>5} | {1:>7} | {2:>5} |'.format(pos_tag_after[0] + pos_tag_after[2],
                                                           pos_tag_after[1] + pos_tag_after[3],
                                                           sum(pos_tag_after))

