import datetime
import re
from lxml import etree

from haystack import indexes
from django.template import loader, Context
from django.utils.html import strip_tags

from axel.articles.models import Article


class PDFCleaner:
    """Utility for cleaning PDF extracted data"""

    # Update this dict as needed
    _CHAR_REPLACEMENT = {
        # latin-1 characters that don't have a unicode decomposition
        0x2012: u'-', # FIGURE DASH
        0x2013: u'-', # EN DASH
        0x2014: u'-', # EM DASH
        0x2212: u'-',
        0x2018: u"'", # LEFT SINGLE QUOTATION MARK
        0x2019: u"'", # RIGHT SINGLE QUOTATION MARK
        0x201c: u'"', # LEFT DOUBLE QUOTATION MARK
        0x201d: u'"', # RIGHT DOUBLE QUOTATION MARK
        0xFB01: u'fi',
        0xFB02: u'fl',
        0xFB03: u'ffi',
        0xFB04: u'ffl',
        0xFB06: u'st',
        0xFB00: u'ff',
        0xFFFD: u''
        }

    _NEW_LINE_REGEX = re.compile(r'-\n.*?<p>', re.S)
    _STRIP_EMPTY_TAGS=re.compile(r'<[a-z]+/>')

    @classmethod
    def _extract_abstract(cls, contents):
        """
        Try to find abstract
        :type contents: list
        :param contents: list of strings
        :return: extracted abstract, if any, index of abstract end
        """
        abs_start = 0
        abs_end = 0
        abstract = ''
        for i, line in enumerate(contents[:20]):
            if line.lower().startswith('abstract'):
                abs_start = i
                break
        if abs_start:
            # detect abstract end
            for i, line in enumerate(contents[abs_start:abs_start+50]):
                if line.lower() == 'introduction' or line.lower() == '1 introduction':
                    abs_end = abs_start + i
                    break
            # fill the abstract if found
        if abs_start and abs_end:
            abstract = ' '.join(contents[abs_start:abs_end])[8:]

        return abstract, abs_end

    @classmethod
    def clean_pdf_data(cls, contents):
        """
        Cleans pdf contents extracted by Solr, tries to extract abstract and title.
        :type contents: str
        :return: dict of text, abstract and title
        """
        result_dict = {'text': '', 'abstract': '', 'title': ''}
        contents = contents.translate(cls._CHAR_REPLACEMENT)
        contents = cls._NEW_LINE_REGEX.sub('', contents)
        contents = cls._STRIP_EMPTY_TAGS.sub('', contents)
        # otherwise etree can't parse it
        contents = contents.replace('encoding="UTF-8"', '')

        #extract body which is article text
        body = etree.fromstring(contents).find('.//{http://www.w3.org/1999/xhtml}body')
        contents = etree.tostring(body)
        contents = strip_tags(contents).strip().split('\n')
        contents = [line for line in contents if line]

        result_dict['title'] = contents[0].strip()
        del contents[0]

        result_dict['abstract'], abs_end_index = cls._extract_abstract(contents)

        # remove abstract from rest of the pdf
        contents = contents[abs_end_index:]
        for i in range(len(contents)):
            if len(contents[i].split())<=2:
                contents[i]+='\n'
        result_dict['text'] = ' '.join(contents)
        return result_dict


class ArticleIndex(indexes.RealTimeSearchIndex, indexes.Indexable):
    """Article indexer for haystack"""
    text = indexes.CharField(document=True, use_template=True)
    abstract = indexes.CharField(model_attr='abstract')
    pub_year = indexes.IntegerField(model_attr='year')

    def get_model(self):
        """returns underlying model"""
        return Article

    def index_queryset(self):
        """Used when the entire index for model is updated."""
        return self.get_model().objects.filter(year__lte=datetime.datetime.now())

    def prepare(self, obj):
        """
        Extract PDF contents and meta-data
        :type obj: Article
        """
        data = super(ArticleIndex, self).prepare(obj)

        # This could also be a regular Python open() call, a StringIO instance
        # or the result of opening a URL. Note that due to a library limitation
        # file_obj must have a .name attribute even if you need to set one
        # manually before calling extract_file_contents:
        obj.pdf.open()
        extracted_data = self._get_backend(None).extract_file_contents(obj.pdf.file)
        result = PDFCleaner.clean_pdf_data(extracted_data['contents'])
        obj.pdf.close()
        if result['abstract']:
            obj.abstract = result['abstract']
        if result['title']:
            obj.title = result['title']
        # save raw because we don't want to trigger signal again
        obj.save_base(raw=True)

        # Now we'll finally perform the template processing to render the
        # text field with *all* of our metadata visible for templating:
        t = loader.select_template(('search/indexes/articles/article_text.txt', ))
        data['text'] = t.render(Context({'object': obj,
                                         'extracted': result}))
        return data

