"""Import PDF files to the database"""
import os
from optparse import make_option

from django.core.management.base import BaseCommand, CommandError
from django.core.files import File
from axel.articles.models import Article, Venue


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
        for root, dirs, files in os.walk(dir):
            for name in files:
                if name.endswith('.pdf'):
                    full_path = os.path.join(root, name)
                    article = Article(venue=venue, year=year)
                    with open(full_path, 'rb') as pdf:
                        article.pdf.save(name, File(pdf), save=True)
                    article.save()
