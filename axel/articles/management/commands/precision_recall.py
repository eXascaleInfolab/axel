"""Match extracted collocation with DBPedia entities"""
from __future__ import division
from collections import Counter, defaultdict
from optparse import make_option
from termcolor import colored

from django.core.management.base import BaseCommand, CommandError

from axel.articles.models import Article
from axel.libs.utils import print_progress


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--cluster', '-c',
                    action='store',
                    dest='cluster',
                    help='cluster id for article type'),
    )
    args = '<method1> <method2> ...'
    help = 'Produce match between collocation and DBPeadia concepts'

    def handle(self, *args, **options):

        self.cluster_id = cluster_id = options['cluster']
        if not cluster_id:
            raise CommandError("need to specify cluster id")
        self.Model = Model = Article.objects.filter(cluster_id=cluster_id)[0].CollocationModel

        correct_objects = set(Model.objects.filter(tags__is_relevant=True).values_list('ngram', flat=True))
        incorrect_objects = set(Model.objects.filter(tags__is_relevant=False).values_list('ngram', flat=True))

        for arg in args:
            eval_method = getattr(self, '_' + arg + '_calculation')
            eval_method(correct_objects, incorrect_objects)

    def _wikilinks_calculation(self, correct_objects, incorrect_objects):
        print 'Generating wikilinks Histogram'
        import requests
        import json
        import re
        template_query = 'http://en.wikipedia.org/w/api.php?action=query&titles={0}&prop=links&plnamespace=0&pllimit=500&format=json'

        relation_distibution = {'valid': 0, 'invalid': 0, 'multi': 0}

        dbpedia_ngrams = set()
        for colloc in self.Model.objects.all():
            if 'dbpedia' in colloc.source:
                dbpedia_ngrams.add(colloc.ngram)

        links_dict = {}

        def _get_links(ngram):
            ngram_links = json.loads(requests.get(template_query.format(ngram)).text)
            try:
                ngram_links = ngram_links['query']['pages'].values()[0]['links']
            except KeyError:
                return []
            ngram_links = [re.sub(r' \(.+\)', '', link['title'].lower()) for link in ngram_links]
            ngram_links = set([ngram for ngram in ngram_links if len(ngram.split()) > 1])
            return ngram_links

        for obj in Article.objects.filter(cluster_id=self.cluster_id):
            print obj
            article = obj
            """:type: Article"""

            article_ngrams = set(article.articlecollocation_set.values_list('ngram', flat=True))
            article_ngrams = list(article_ngrams.intersection(dbpedia_ngrams))

            for i, ngram1 in enumerate(article_ngrams):
                if ngram1 in links_dict:
                    ngram1_links = links_dict[ngram1]
                else:
                    ngram1_links = _get_links(ngram1)
                    links_dict[ngram1] = ngram1_links
                for j in range(i+1, len(article_ngrams)):
                    ngram2 = article_ngrams[j]
                    if ngram2 in links_dict:
                        ngram2_links = links_dict[ngram2]
                    else:
                        ngram2_links = _get_links(ngram2)
                        links_dict[ngram2] = ngram2_links

                    if ngram1 in correct_objects and ngram2 in correct_objects:
                        attr = 'valid'
                    elif ngram1 in incorrect_objects and ngram2 in incorrect_objects:
                        attr = 'invalid'
                    else:
                        attr = 'multi'
                    if ngram1 in ngram2_links or ngram2 in ngram1_links:
                        relation_distibution[attr] += 1
        print relation_distibution


    def _dbpedia_cc_size_calculation(self, correct_objects, incorrect_objects):
        print 'Generating DBPedia Connected Component size Histogram'
        import networkx as nx

        cc_size_distibution = {'valid': defaultdict(lambda: 0), 'invalid': defaultdict(lambda: 0)}

        for obj in Article.objects.filter(cluster_id=self.cluster_id):
            print obj
            article = obj
            """:type: Article"""
            graph = article.dbpedia_graph
            results = []
            for component in nx.connected_components(graph):
                component = [node for node in component if 'Category' not in node]
                results.append(component)
                conn_valid = len([node for node in component if node in correct_objects])
                conn_invalid = len([node for node in component if node in incorrect_objects])
                if conn_valid > conn_invalid:
                    cc_size_distibution['valid'][len(component)] += 1
                else:
                    cc_size_distibution['invalid'][len(component)] += 1
        print cc_size_distibution


    def _dbpedia_calculation(self, correct_objects, incorrect_objects):
        print 'Calculating Precision/Recall using dbpedia graph method'
        precision = []
        recall = []

        dbpedia_ngrams = set()
        for colloc in self.Model.objects.all():
            if 'dbpedia' in colloc.source:
                dbpedia_ngrams.add(colloc.ngram)

        top_false_counter = Counter()
        for obj in Article.objects.filter(cluster_id=self.cluster_id):
            article = obj
            """:type: Article"""
            results = article.dbpedia_graph
            article_ngrams = set(article.articlecollocation_set.values_list('ngram', flat=True))
            results_all = article_ngrams.intersection(dbpedia_ngrams)

            true_pos = [x for x in results if x in correct_objects]
            good_removed = [x for x in results_all if x in correct_objects and x not in true_pos]
            false_pos = [x for x in results if x in incorrect_objects]
            top_false_counter.update(false_pos)
            local_precision = len(true_pos) / len([x for x in results if x in correct_objects or
                                                                         x in incorrect_objects])
            local_recall = len(true_pos) / len([x for x in article.articlecollocation_set
            .values_list('ngram', flat=True) if x in correct_objects])

            precision.append(local_precision)
            recall.append(local_recall)
            print obj.id
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

    def _dblp_calculation(self, correct_objects, incorrect_objects):
        print 'Calculating Precision/Recall using DBLP match'
        precision = []
        recall = []
        dblp_ngrams = set()
        for colloc in self.Model.objects.all():
            if 'dblp' in colloc.source:
                dblp_ngrams.add(colloc.ngram)
        #print dblp_ngrams

        for obj in Article.objects.filter(cluster_id=self.cluster_id):
            article = obj
            """:type: Article"""
            article_ngrams = set(article.articlecollocation_set.values_list('ngram', flat=True))
            results = article_ngrams.intersection(dblp_ngrams)
            true_pos = [x for x in results if x in correct_objects]
            false_pos = [x for x in results if x not in incorrect_objects]
            print obj.id
            print colored(true_pos, 'green')
            print colored(false_pos, 'red')
            print
            local_precision = len(true_pos) / len([x for x in results if x in correct_objects or
                                                                         x in incorrect_objects])
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

