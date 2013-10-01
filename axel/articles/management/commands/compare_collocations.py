"""Compares two different collocation candidate sets"""
from __future__ import division
from collections import defaultdict
from optparse import make_option
from termcolor import colored

from django.core.management.base import BaseCommand, CommandError
from axel.articles.models import Article, TestCollocations, CLUSTERS_DICT


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--cluster', '-c', action='store', dest='cluster',
                    help='cluster name'),
        make_option('--dry-run',
                    action='store_true',
                    dest='dry',
                    default=False,
                    help='Only print the results'),
    )

    help = 'Compares two different collocation candidate sets'

    def handle(self, *args, **options):
        cluster = options['cluster']
        if not cluster:
            raise CommandError("need to specify cluster")

        dry = options['dry']

        model = CLUSTERS_DICT[cluster]

        self.article_rel_dict = defaultdict(lambda: {0: set(), 1: set()})
        for key, is_rel in model.judged_data.iteritems():
            ngram, article_id = key.split(',')
            is_rel = int(is_rel)
            self.article_rel_dict[article_id][is_rel].add(ngram)

        # precision-recall
        old_prec = []
        old_rec = []
        new_prec = []
        new_rec = []

        for article in Article.objects.filter(cluster_id=cluster):
            print article, article.pdf
            test_collocs = list(TestCollocations.objects.filter(article=article).values_list('ngram', 'count'))
            cur_collocs = list(model.objects.filter(article=article).values_list('ngram', 'count'))

            # START: calculate precision-recall
            correct_objects = self.article_rel_dict[unicode(article)][1]
            incorrect_objects = self.article_rel_dict[unicode(article)][0]
            true_pos_old = [x for x in zip(*cur_collocs)[0] if x in correct_objects]
            true_pos_new = [x for x in zip(*test_collocs)[0] if x in correct_objects]
            false_pos_old = [x for x in zip(*cur_collocs)[0] if x in incorrect_objects]
            false_pos_new = [x for x in zip(*test_collocs)[0] if x in incorrect_objects]
            local_old_prec = len(true_pos_old) / (len(true_pos_old) + len(false_pos_old))
            local_new_prec = len(true_pos_new) / (len(true_pos_new) + len(false_pos_new))
            print 'Obsolote correct:'
            print colored(set(true_pos_old).difference(true_pos_new), 'green')
            old_prec.append(local_old_prec)
            new_prec.append(local_new_prec)
            old_rec_local = len(true_pos_old) / len(correct_objects)
            new_rec_local = len(true_pos_new) / len(correct_objects)
            old_rec.append(old_rec_local)
            new_rec.append(new_rec_local)
            # END: calculate precision-recall

            obsolete_collocs = set(zip(*cur_collocs)[0]).difference(zip(*test_collocs)[0])
            print 'Obsolete collocations:'
            print obsolete_collocs
            if not dry:
                model.objects.filter(article=article, ngram__in=obsolete_collocs).delete()
            new_collocs = set(zip(*test_collocs)[0]).difference(zip(*cur_collocs)[0])
            print 'New collocations:'
            print new_collocs
            if not dry:
                for ngram, count in test_collocs:
                    if ngram in new_collocs:
                        obj = model(ngram=ngram, count=count, article=article, total_count=count,
                                    extra_fields={})
                        obj.save()
                        print obj.pos_tag, obj.pos_tag_prev, obj.pos_tag_after
            print

        precision_old = sum(old_prec) / len(old_prec)
        precision_new = sum(new_prec) / len(new_prec)
        recall_old = sum(old_rec) / len(old_rec)
        recall_new = sum(new_rec) / len(new_rec)

        print 'Current Precision: ', precision_old
        print 'New Precision: ', precision_new
        print 'Current Recall: ', recall_old
        print 'New Recall: ', recall_new

