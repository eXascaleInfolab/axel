"""Get edits statistics for the files"""
import re

from collections import Counter
from django.core.management.base import BaseCommand

import difflib

SENTENCE_REGEX = re.compile(r'[.!?]')


class Command(BaseCommand):
    args = '<file1 file2 ...>'
    help = 'Analyze and display edit statistics for the files' \
           '"\\n" is used for external separators, and ";;;" for internal separators'

    def _get_sentence(self, text, sentences, index1):
        for i, index in enumerate(sentences):
            if index1 > index:
                continue
            if i == 0:
                return text[:index].strip()
            else:
                return text[sentences[i-1]+1:index].strip()

    def handle(self, *args, **options):
        insertCounter = {}
        deleteCounter = {}

        for edit_file in args:
            for line in open(edit_file).read().split('\n'):
                line = line.strip().replace('\\r\\n', '\n')
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

                            # skip edits that contain line breaks and sentence seprators
                            if set('.!?\n') & set(str1 + str2):
                                continue
                            sentence1 = self._get_sentence(edit1, sentences1, i1)
                            sentence2 = self._get_sentence(edit2, sentences2, j1)

                            if tag == 'replace':
                                pass
                                #insertCounter.append(str2)
                                #deleteCounter.append(str1)
                            elif tag == 'delete':
                                deleteCounter[str1] = (sentence1, sentence2, 'delete')
                            elif tag == 'insert':
                                insertCounter[str2] = (sentence1, sentence2, 'insert')
        insertKeys = [x for x in Counter(insertCounter.keys()).iteritems() if x[1] > 1]
        deleteKeys = [x for x in Counter(deleteCounter.keys()).iteritems() if x[1] > 1]
