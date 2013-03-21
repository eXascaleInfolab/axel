"""Match extracted collocation with DBPedia entities"""
from __future__ import division
from optparse import make_option

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

    def _dbpedia_calculation(self, correct_objects, incorrect_objects):
        print 'Calculating Precision/Recall using dbpedia graph method'
        precision = []
        recall = []
        for obj in print_progress(Article.objects.filter(cluster_id=self.cluster_id)):
            article = obj
            """:type: Article"""
            results = article.dbpedia_graph
            true_pos = [x for x in results if x in correct_objects]
            local_precision = len(true_pos) / len([x for x in results if x in correct_objects or
                                                                         x in incorrect_objects])
            local_recall = len(true_pos) / len([x for x in article.articlecollocation_set
            .values_list('ngram', flat=True) if x in correct_objects])

            precision.append(local_precision)
            recall.append(local_recall)

        print precision
        print recall
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        print 'Precision: ', precision
        print 'Recall', recall
        print 'F1 measure', 2 * (precision * recall) / (precision + recall)

    def _dblp_calculation(self, correct_objects, incorrect_objects):
        print 'Calculating Precision/Recall using DBLP match'
        precision = []
        recall = []
        dblp_ngrams = set()
        for colloc in self.Model.objects.all():
            if 'dblp' in colloc.source:
                dblp_ngrams.add(colloc.ngram)
        print dblp_ngrams

        for obj in print_progress(Article.objects.filter(cluster_id=self.cluster_id)):
            article = obj
            """:type: Article"""
            article_ngrams = set(article.articlecollocation_set.values_list('ngram', flat=True))
            results = article_ngrams.intersection(dblp_ngrams)
            true_pos = [x for x in results if x in correct_objects]
            local_precision = len(true_pos) / len([x for x in results if x in correct_objects or
                                                                         x in incorrect_objects])
            local_recall = len(true_pos) / len([x for x in article_ngrams if x in correct_objects])

            precision.append(local_precision)
            recall.append(local_recall)

        print precision
        print recall
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        print 'Precision: ', precision
        print 'Recall', recall
        print 'F1 measure', 2 * (precision * recall) / (precision + recall)
