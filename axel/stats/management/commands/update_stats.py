"""Match extracted collocation with DBPedia entities"""
from __future__ import division
from collections import defaultdict
from optparse import make_option

from django.core.management.base import BaseCommand, CommandError

from axel.articles.models import CLUSTERS_DICT
from axel.stats.models import STATS_CLUSTERS_DICT
from axel.libs.utils import print_progress


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--cluster', '-c',
                    action='store',
                    dest='cluster',
                    help='cluster id for article type'),
    )

    help = 'Updates aggregated statistics for the specified cluster, like max POS tag, total counts'

    def handle(self, *args, **options):

        self.cluster_id = cluster_id = options['cluster']
        if not cluster_id:
            raise CommandError("need to specify cluster id")
        self.Model = CLUSTERS_DICT[cluster_id].objects
        self.StatsModel = STATS_CLUSTERS_DICT[cluster_id].objects
        #self._update_total_counts()
        #self._update_max_pos_tags()
        self._add_delete_stats()

    def _update_total_counts(self):
        print 'Update total counts:'
        self.StatsModel.all().update(count=0)
        total_counts = defaultdict(lambda: 0)
        print 'Collecting new counts...'
        for ngram, count in print_progress(self.Model.values_list('ngram', 'count'), 10):
            total_counts[ngram] += count
        print 'Updating total counts...'
        for ngram, count in print_progress(total_counts.items(), 5):
            self.StatsModel.filter(ngram=ngram).update(count=count)
            self.Model.filter(ngram=ngram).update(total_count=count)

    def _update_max_pos_tags(self):
        print 'Update max POS tags'
        self.StatsModel.all().update(_max_pos_tag=None)
        for c in print_progress(self.StatsModel.all(), 5):
            _ = c.max_pos_tag

    def _add_delete_stats(self):
        cur_ngrams = set(self.Model.values_list('ngram', flat=True))
        cur_stat_ngrams = set(self.StatsModel.values_list('ngram', flat=True))
        print 'New ngrams:'
        new_ngrams = cur_ngrams.difference(cur_stat_ngrams)
        print new_ngrams
        if new_ngrams:
            answer = raw_input('Create new? (y/n): ')
            if answer == 'y':
                for ngram in new_ngrams:
                    count = sum(self.Model.filter(ngram=ngram).values_list('count', flat=True))
                    stat_ngram = self.StatsModel.create(ngram=ngram, count=count)
                    _ = stat_ngram.max_pos_tag
                    self.Model.filter(ngram=ngram).update(total_count=count)
                print 'Created'

        print 'Obsolete ngrams:'
        obsolete = cur_stat_ngrams.difference(cur_ngrams)
        print obsolete
        if obsolete:
            answer = raw_input('Delete? (y/n): ')
            if answer == 'y':
                self.StatsModel.filter(ngram__in=obsolete).delete()
                print 'Deleted'
