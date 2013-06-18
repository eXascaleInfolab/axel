"""Get edits statistics for the files"""
import re

from collections import Counter
from django.core.management.base import BaseCommand

import difflib
from axel.ngrams.models import Sentence, Edit

SENTENCE_REGEX = re.compile(r'[.!?]')
FILTER_REGEX = re.compile(r'[^\w\s\d]')
LINE_END_REPLACE_REGEX = re.compile(r'(?P<end>[.?!])(\\r\\n)+')
MULTIPLE_PUNCT_REGEX = re.compile(r'([.!?])+')


class Command(BaseCommand):
    args = '<file1 file2 ...>'
    help = 'Analyze and display edit statistics for the files' \
           '"\\n" is used for external separators, and ";;;" for internal separators'

    def _get_sentence(self, text, sentences, index1):
        for i, index in enumerate(sentences):
            if index1 > index:
                continue
            if i == 0:
                return text[:index+1].strip()
            else:
                return text[sentences[i-1]+1:index+1].strip()

        if not sentences:
            return text
        else:
            # we have sentences but the index is bigger
            return text[sentences[-1]+1:]

    def handle(self, *args, **options):
        totalCounter = []

        for edit_file in args:
            for line in open(edit_file).read().split('\n'):
                # treat line breaks as sentences
                line = line.strip()
                line = LINE_END_REPLACE_REGEX.sub('\g<end>', line)
                line = line.replace('\\r\\n', '')
                line = MULTIPLE_PUNCT_REGEX.sub('\g<1>', line)
                if not line or line == "null":
                    continue
                for edit in line.split('|||'):
                    edit1, edit2 = edit[1:-1].split(';;;')
                    sentences1 = [c.start() for c in SENTENCE_REGEX.finditer(edit1)]
                    sentences2 = [c.start() for c in SENTENCE_REGEX.finditer(edit2)]
                    for seq in difflib.SequenceMatcher(None, edit1, edit2).get_grouped_opcodes(0):
                        for tag, i1, i2, j1, j2 in seq:
                            if tag == 'equal':
                                continue
                            str1, str2 = edit1[i1:i2], edit2[j1:j2]

                            # skip edits that contain line breaks and sentence separators
                            if set('.!?\n') & set(str1 + str2):
                                continue

                            # skip bad edits:
                            if FILTER_REGEX.search(str1) or FILTER_REGEX.search(str2):
                                continue

                            sentence1 = self._get_sentence(edit1, sentences1, i1)
                            sentence2 = self._get_sentence(edit2, sentences2, j1)

                            if tag == 'replace':
                                pass
                                #insertCounter.append(str2)
                                #deleteCounter.append(str1)
                            elif tag == 'delete':
                                totalCounter.append((str1, sentence1, sentence2, Edit.DELETE))
                            elif tag == 'insert':
                                totalCounter.append((str2, sentence1, sentence2, Edit.INSERT))
        totalKeys = set([x[0] for x in Counter(zip(*totalCounter)[0]).iteritems() if x[1] > 1])

        for edit, sentence1, sentence2, edit_type in totalCounter:
            if edit in totalKeys:
                try:
                    sen = Sentence.objects.get(sentence1=sentence1, sentence2=sentence2)
                except Sentence.DoesNotExist:
                    sen = Sentence.objects.create(sentence1=sentence1, sentence2=sentence2)
                Edit.objects.create(sentence=sen, edit_type=edit_type, edit1=edit)
