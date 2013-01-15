"""Import PDF files to the database"""
import pickle
from django.utils.html import strip_tags
from lxml import etree
from optparse import make_option

from django.core.management.base import BaseCommand, CommandError

from axel.articles.utils import nlp
from axel.articles.utils.nlp import Stemmer, build_ngram_index


class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--article', '-a',
            action='store',
            dest='article_id',
            help='article id'),
        )
    help = 'Extracts concepts from different representations'

    def handle(self, *args, **options):
        article_id = options['article_id']
        if not article_id:
            raise CommandError("need to specify article id")

        # Extract concepts
        pdfx_file = '{0}.pdfx.xml'.format(article_id)
        pdf_file = '{0}.pdf.txt'.format(article_id)
        texpp_file = '{0}.texpp.txt'.format(article_id)
        import codecs

        pdfx_text = codecs.open(pdfx_file, 'r', 'utf-8').read()
        contents = pdfx_text.replace('encoding=\'UTF-8\'', '')
        body = etree.fromstring(contents).find('.//body')
        contents = etree.tostring(body)
        contents = strip_tags(contents).strip().split('\n')
        contents = [line for line in contents if line]
        pdfx_text = ' '.join(contents)
        pdfx_text = Stemmer.stem_wordnet(pdfx_text)

        pdf_text = nlp.stem_text(codecs.open(pdf_file, 'r', 'utf-8').read())
        pdf_text = pdf_text['abstract']+'\n'+pdf_text['text']
        texpp_text = Stemmer.stem_wordnet(codecs.open(texpp_file, 'r', 'utf-8').read())

        ontology = pickle.load(open('ontology.pcl'))
        ontology = set([item for subl in ontology.values() for item in subl])

        comparing_counts = []
        for text in (pdf_text, pdfx_text, texpp_text):
            #build n-gram index for each result
            index = build_ngram_index(text)
            all_concepts = list(ontology.intersection(index.keys()))
            all_concepts.sort(key=lambda x: len(x))
            concept_counts = {}
            for c in all_concepts:
                concept_counts[c] = index[c]

            # normalize quantities
            for c in reversed(all_concepts):
                if concept_counts[c] == 0:
                    continue
                for c1 in sorted(build_ngram_index(c).keys(), key=lambda x: len(x))[:-1]:
                    if c1 in concept_counts:
                        concept_counts[c1] -= concept_counts[c]

            concept_counts = dict([c for c in concept_counts.items() if c[1]>0])
            comparing_counts.append(set(concept_counts.keys()))

        common_concepts = set(comparing_counts[0]).intersection(comparing_counts[1]).intersection(comparing_counts[2])
        print len(common_concepts)

        for i, concepts in enumerate(comparing_counts):
            comparing_counts[i] = set(comparing_counts[i]).difference(common_concepts)
        print comparing_counts
