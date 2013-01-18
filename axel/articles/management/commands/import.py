"""Import PDF files to the database"""
import json
import os
from optparse import make_option

from django.core.management.base import BaseCommand, CommandError
from django.core.files import File
from axel.articles.models import Article, Venue, ArticleCollocation
from axel.articles.utils.nlp import _update_ngram_counts


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--directory', '-d',
            action='store',
            dest='dir',
            help='directory to import PDFs from'),
        make_option('--venue', '-c', action='store', dest='venue',
            help='Conference name, one of: {0}'.format(', '.join(Venue.objects.values_list(
                'acronym', flat=True)))),
        make_option('--year', '-y', action='store', dest='year',
            help='Conference year')
        )
    help = 'Imports PDFs from the specified directory'

    def handle(self, *args, **options):
        dir = options['dir']
        venue = options['venue']
        year = int(options['year'])
        if not dir:
            raise CommandError("need to specify directory")
        if not venue:
            raise CommandError("need to specify venue")
        if not year:
            raise CommandError("need to specify year")

        venue = Venue.objects.get(acronym=venue)

        # Traverse and import PDFs
        article_ids = []
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith('.pdf'):
                    full_path = os.path.join(root, name)
                    article = Article(venue=venue, year=year)
                    with open(full_path, 'rb') as pdf:
                        article.pdf.save(name, File(pdf), save=True)
                    article.save()
                    article_ids.append(article.id)

        print 'Starting updates...'
        for article in Article.objects.filter(id__in=article_ids):
            ngrams = sorted(article.articlecollocation_set.values_list('keywords','count'),
                                                                key=lambda x:(x[1],x[0]))
            new_ngrams = _update_ngram_counts([c.split() for c in zip(*ngrams)[0]],
                json.loads(article.index))
            new_ngrams = sorted(new_ngrams.items(),key=lambda x:(x[1],x[0]))
            new_ngrams = [k for k in new_ngrams if k[1]>0]
            if new_ngrams != ngrams:
                obsolete_ngrams = set(ngrams).difference(new_ngrams)
                article.articlecollocation_set.filter(keywords__in=zip(*obsolete_ngrams)[0])\
                                                                                        .delete()
                for ngram, score in set(new_ngrams).difference(ngrams):
                    ArticleCollocation.objects.create(keywords=ngram, count=score, article=article)


