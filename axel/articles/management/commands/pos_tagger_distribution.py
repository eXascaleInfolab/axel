"""Output POS tagger distributions for highcharts histogram"""
from collections import defaultdict
from optparse import make_option
from django.contrib.contenttypes.models import ContentType
from django.core.management import BaseCommand, CommandError
from django.db.models.loading import get_model

import nltk
import re
from test_collection.models import TaggedCollection
from axel.libs.utils import print_progress, get_contexts_ngrams


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--model', '-m',
            action='store',
            dest='model',
            help='model to tag POS from'),
        )
    help = 'POS tagger statistics'

    def _compress_pos_tag(self, max_ngram_tags):
        max_ngram = ' '.join(max_ngram_tags)
        if max_ngram.startswith('CD '):
            max_ngram = 'NUM XXX'
        elif re.search('VB\w$', max_ngram):
            max_ngram = 'XXX VERB'
        elif max_ngram.startswith('RB '):
            max_ngram = 'ADV XXX'
        elif re.search(r'^VB[^GN]', max_ngram):
            max_ngram = 'VERB XXX'
        return max_ngram

    def handle(self, *args, **options):
        if not options['model']:
            raise CommandError("need to specify model")
        app_label, model = options['model'].split('.')[1::2]
        Model = get_model(app_label, model)
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
            contexts = ngram.all_contexts(func=get_contexts_ngrams)
            if not contexts:
                contexts = [(ngram.ngram, ngram.ngram)]
            ngram_len = len(ngram.ngram.split())
            for i, context in enumerate(contexts):
                words, context = context
                words = tuple(words.split())
                tags = [(word, tag) for word, tag in nltk.pos_tag(nltk.word_tokenize(context)) if
                        word in set(words)]
                found = False
                for j, wordtag in enumerate(tags):
                    if wordtag[0] == words[0] and tuple(zip(*tags)[0][j:j+ngram_len]) == words:
                        tags = tuple(zip(*tags)[1][j:j+ngram_len])
                        found = True
                        break
                if not found:
                    print context, ngram.ngram
                    continue

                ngram_tags[tags] += 1
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
            max_ngram = self._compress_pos_tag(max_ngram_tags)
            all_tags.add(max_ngram)
            results[int(obj.is_relevant)][max_ngram] += 1
        all_tags = sorted(all_tags)
        relevant = [results[1][tag] for tag in all_tags]
        irrelevant = [results[0][tag] for tag in all_tags]
        print results
        print all_tags
        print 'Relevant:', relevant
        print 'Irrelevant', irrelevant
