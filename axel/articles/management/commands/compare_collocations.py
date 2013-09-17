"""Compares two different collocation candidate sets"""
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

        for article in Article.objects.filter(cluster_id=cluster):
            print article, article.pdf
            test_collocs = list(TestCollocations.objects.filter(article=article).values_list('ngram', 'count'))
            cur_collocs = list(model.objects.filter(article=article).values_list('ngram', 'count'))
            obsolete_collocs = set(zip(*cur_collocs)[0]).difference(zip(*test_collocs)[0])
            print colored('Obsolete collocations:', 'red')
            print colored(obsolete_collocs, 'red')
            if not dry:
                model.objects.filter(article=article, ngram__in=obsolete_collocs).delete()
            new_collocs = set(zip(*test_collocs)[0]).difference(zip(*cur_collocs)[0])
            print colored('New collocations:', 'green')
            print colored(new_collocs, 'green')
            if not dry:
                for ngram, count in test_collocs:
                    if ngram in new_collocs:
                        obj = model(ngram=ngram, count=count, article=article, total_count=count,
                                    extra_fields={})
                        obj.save()
                        print obj.pos_tag, obj.pos_tag_prev, obj.pos_tag_after
            print

