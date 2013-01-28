"""Import PDF files to the database"""
import json
from django.contrib.contenttypes.models import ContentType
import os
from optparse import make_option

from django.core.management.base import BaseCommand, CommandError
from django.core.files import File
from test_collection.models import TaggedCollection
from axel.articles.models import Article, Venue, ArticleCollocation
from axel.articles.utils.nlp import _update_ngram_counts
from axel.libs.utils import print_progress
from axel.stats.models import SWCollocations


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--directory', '-d',
            action='store',
            dest='dir',
            help='directory to import PDFs from'),
        make_option('--venue', action='store', dest='venue',
            help='Conference name, one of: {0}'.format(', '.join(Venue.objects.values_list(
                'acronym', flat=True)))),
        make_option('--year', '-y', action='store', dest='year',
            help='Conference year'),
        make_option('--cluster', '-c', action='store', dest='cluster',
            help='cluster name')
        )

    help = 'Imports PDFs from the specified directory'

    def handle(self, *args, **options):
        dir = options['dir']
        venue = options['venue']
        year = int(options['year'])
        cluster = options['cluster']
        if not dir:
            raise CommandError("need to specify directory")
        if not venue:
            raise CommandError("need to specify venue")
        if not year:
            raise CommandError("need to specify year")
        if not cluster:
            raise CommandError("need to specify cluster")

        venue = Venue.objects.get(acronym=venue)

        # Traverse and import PDFs
        article_ids = []
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith('.pdf'):
                    full_path = os.path.join(root, name)
                    article = Article(venue=venue, year=year, cluster_id=cluster)
                    with open(full_path, 'rb') as pdf:
                        article.pdf.save(name, File(pdf), save=True)
                    article.save()
                    article_ids.append(article.id)

        print 'Starting updates...'
        for article in print_progress(Article.objects.filter(id__in=article_ids)):
            ngrams = sorted(article.articlecollocation_set.values_list('ngram','count'),
                                                                key=lambda x:(x[1],x[0]))
            new_ngrams = _update_ngram_counts([c.split() for c in zip(*ngrams)[0]],
                json.loads(article.index))
            new_ngrams = sorted(new_ngrams.items(),key=lambda x:(x[1],x[0]))
            new_ngrams = [k for k in new_ngrams if k[1]>0]
            if new_ngrams != ngrams:
                obsolete_ngrams = set(ngrams).difference(new_ngrams)
                article.articlecollocation_set.filter(ngram__in=zip(*obsolete_ngrams)[0])\
                                                                                        .delete()
                for ngram, score in set(new_ngrams).difference(ngrams):
                    ArticleCollocation.objects.create(ngram=ngram, count=score, article=article)

        print 'Starting merging... (dashed ngrams)'
        all_ngrams = set(ArticleCollocation.objects.values_list('ngram', flat=True).distinct())
        dashed_ngrams = [ngram for ngram in all_ngrams if '-' in ngram]
        for d_ngram in dashed_ngrams:
            if d_ngram.replace('-', ' ') in all_ngrams:
                print d_ngram
                # TODO: MEGRE
        #SWCollocations.objects.filter(count=0).delete()

#        # mark correct from the ontology
#        import pickle
#        ontology = pickle.load(open('ontology.pcl'))
#        ontology = set([item for subl in ontology.values() for item in subl])
#        ct = ContentType.objects.get_for_model(SWCollocations)
#        for c in SWCollocations.objects.all():
#            if c.ngram in ontology or c.ngram.replace('-', ' ') in ontology:
#                TaggedCollection.objects.get_or_create(object_id=c.id,content_type=ct,is_relevant=True)
