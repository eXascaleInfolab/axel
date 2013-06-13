"""Get edits statistics for the files"""
from collections import Counter
from django.core.management.base import BaseCommand

import difflib


class Command(BaseCommand):
    args = '<file1 file2 ...>'
    help = 'Analyze and display edit statistics for the files' \
           '"\\n" is used for external separators, and ";;;" for internal separators'

    def handle(self, *args, **options):
        insertCounter = []
        deleteCounter = []

        for edit_file in args:
            for line in open(edit_file).read().split('\n'):
                line = line.strip().replace('\\r\\n', '\n')
                if not line or line == "null":
                    continue
                for edit in line.split('|||'):
                    edit1, edit2 = edit[1:-1].split(';;;')
                    for seq in difflib.SequenceMatcher(None, edit1, edit2).get_grouped_opcodes(0):
                        for tag, i1, i2, j1, j2 in seq:
                            if tag == 'equal':
                                continue
                            str1, str2 = edit1[i1:i2], edit2[j1:j2]

                            if tag == 'replace':
                                pass
                                #insertCounter.append(str2)
                                #deleteCounter.append(str1)
                            elif tag == 'delete':
                                deleteCounter.append(str1)
                            elif tag == 'insert':
                                insertCounter.append(str2)
        print [x for x in Counter(insertCounter).iteritems() if x[1] > 1]
        print [x for x in Counter(deleteCounter).iteritems() if x[1] > 1]
