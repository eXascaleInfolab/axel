"""Output POS tagger distributions for highcharts histogram"""
from collections import defaultdict
from optparse import make_option
from django.contrib.contenttypes.models import ContentType
from django.core.management import BaseCommand, CommandError
from django.db.models.loading import get_model

import nltk
import re
from test_collection.models import TaggedCollection
from axel.libs.utils import print_progress

class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--model', '-m',
            action='store',
            dest='model',
            help='model to tag POS from'),
        )
    help = 'POS tagger statistics'

    def handle(self, *args, **options):
        if not options['model']:
            raise CommandError("need to specify model")
        app_label, model = options['model'].rsplit('.', 1)
        Model = get_model('stats', 'SWCollocations')
        ct = ContentType.objects.get_for_model(Model)
        results = [defaultdict(lambda:0), defaultdict(lambda:0)]
        all_tags = set()
        for obj in print_progress(TaggedCollection.objects.filter(content_type=ct)
                    .select_related('object')):
            ngram_tags = defaultdict(lambda: 0)
            ngram = obj.object
            """:type: Collocation"""
            if not ngram:
                continue
            contexts = ngram.all_contexts
            ngram_len = len(ngram.ngram.split())
            words = set(ngram.ngram.split())
            for i, context in enumerate(contexts):
                tags = [tag for word, tag in nltk.pos_tag(nltk.word_tokenize(context)) if
                        word in words][:ngram_len]
                ngram_tags[tuple(tags)] += 1
                # check every 10 iterations for speed
                if not i % 10 and i > 1:
                    if len(ngram_tags) == 1:
                        break
                    else:
                        items = sorted(ngram_tags.items(), key=lambda x: x[1], reverse=True)
                        if items[0][1] > 5*items[1][1]:
                            break

            # select max weight combination
            max_ngram_tags = max(ngram_tags.items(), key=lambda x: x[1])[0]
            max_ngram = re.sub(r'(VB)\w', r'\1' ,' '.join(max_ngram_tags))
            all_tags.add(max_ngram)
            results[int(obj.is_relevant)][max_ngram] += 1
        all_tags = sorted(all_tags)
        relevant = [results[1][tag] for tag in all_tags]
        irrelevant = [results[0][tag] for tag in all_tags]
        print all_tags
        print 'Relevant:', relevant
        print 'Irrelevant', irrelevant
