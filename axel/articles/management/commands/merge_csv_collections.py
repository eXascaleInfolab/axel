"""Merge two CSV files with judged collections"""
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    args = '<csv_file csv_file ...>'
    help = 'Merge two or more CSV files with judged collection data'

    def handle(self, *args, **options):
        valid_data = set()
        invalid_data = set()
        for csv_file in args:
            for line in open(csv_file).read().split('\n'):
                if not line.strip():
                    continue
                value, is_valid = line.split(',')[:2]
                if is_valid == '1':
                    valid_data.add(value)
                else:
                    invalid_data.add(value)

        result_file = open(args[0], 'w')
        for item in sorted(valid_data):
            result_file.write(item+',1\n')
        for item in sorted(invalid_data):
            result_file.write(item+',0\n')
        result_file.close()
        print 'Merged files successfully. Result written to {0}'.format(args[0])
