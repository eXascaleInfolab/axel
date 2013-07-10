"""Get edits statistics for the files"""
import re

from collections import Counter
from django.core.management.base import BaseCommand

import difflib
from axel.ngrams.models import Sentence, Edit

FILTER_REGEX = re.compile(r'[^\w]')
SENTENCE_REGEX = re.compile(r'\?|(?:([.!])(?:[A-Z]| [A-Z]))')
LINE_END_REPLACE_REGEX = re.compile(r'(?P<end>[.?!])(\\r\\n)+')
MULTIPLE_PUNCT_REGEX = re.compile(r'([.!?])+')


class Command(BaseCommand):
    args = '<file1 file2 ...>'
    help = 'Analyze and display edit statistics for the files' \
           '"\\n" is used for external separators, and ";;;" for internal separators'

    def _get_sentence(self, text, sentences, index1):
        for i, index in enumerate(sentences):
            if index1 >= index:
                continue
            if i == 0:
                return text[:index].strip()
            else:
                return text[sentences[i-1]:index].strip()

        if not sentences:
            return text
        else:
            # we have sentences but the index is bigger
            return text[sentences[-1]:]

    def _debug_check_sentence(self, sentences, edit):
        if sentences:
            i = sentences[0]
            for j in sentences[1:]:
                print edit[i:j]
                i = j
        print

    def handle(self, *args, **options):
        totalCounter = []

        for edit_file in args:
            contents = open(edit_file).read()
            contents.replace("I'm ", 'I am ').replace(" don't", " do not").replace(" doesn't", "does not")
            for line in contents.split('\n'):
                # treat line breaks as sentences
                line = line.strip()
                line = LINE_END_REPLACE_REGEX.sub('\g<end>', line)
                line = line.replace('\\r\\n', '')
                line = MULTIPLE_PUNCT_REGEX.sub('\g<1>', line)
                if not line or line == "null":
                    continue
                # strip quotes
                line = line[1:-1]
                for edit in line.split('|||'):
                    edit1, edit2 = edit.split(';;;')

                    sentences1 = [c.start()+1 for c in SENTENCE_REGEX.finditer(edit1)]
                    sentences2 = [c.start()+1 for c in SENTENCE_REGEX.finditer(edit2)]

                    for seq in difflib.SequenceMatcher(None, edit1, edit2).get_grouped_opcodes(0):
                        for tag, i1, i2, j1, j2 in seq:
                            if tag == 'equal':
                                continue
                            str1, str2 = edit1[i1:i2], edit2[j1:j2]

                            # skip bad edits:
                            if FILTER_REGEX.search(str1) or FILTER_REGEX.search(str2):
                                continue

                            if str1.lower() == str2.lower():
                                continue

                            sentence1 = self._get_sentence(edit1, sentences1, i1)
                            sentence2 = self._get_sentence(edit2, sentences2, j1)

                            # get sentence index and subtract it because index is absolute
                            sentence1_index = edit1.index(sentence1)
                            sentence2_index = edit2.index(sentence2)

                            edit_info = (i1-sentence1_index, i2-sentence1_index,
                                         j1-sentence2_index, j2-sentence2_index)

                            if tag == 'replace':
                                pass
                                #insertCounter.append(str2)
                                #deleteCounter.append(str1)
                            elif tag == 'delete':
                                totalCounter.append((str1, sentence1, sentence2,
                                                     edit_info, Edit.DELETE))
                            elif tag == 'insert':
                                totalCounter.append((str2, sentence1, sentence2,
                                                     edit_info, Edit.INSERT))

        totalKeys = set([x[0] for x in Counter(zip(*totalCounter)[0]).iteritems() if x[1] > 1])

        i = 0

        for edit, sentence1, sentence2, edit_info, edit_type in totalCounter:
            if edit in totalKeys:
                try:
                    sen = Sentence.objects.get(sentence1=sentence1, sentence2=sentence2)
                except Sentence.DoesNotExist:
                    sen = Sentence.objects.create(sentence1=sentence1, sentence2=sentence2)
                Edit.objects.create(sentence=sen, edit_type=edit_type, edit1=edit,
                                    start_pos_orig=edit_info[0], end_pos_orig=edit_info[1],
                                    start_pos_new=edit_info[2], end_pos_new=edit_info[3])
                i += 1
        print "Total edits created:", i
