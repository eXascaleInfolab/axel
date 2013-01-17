"""PDF data extraction"""

import re
from lxml import etree

from django.utils.html import strip_tags
from axel.libs.utils import print_timing


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

    _NEW_LINE_REGEX = re.compile(r'-\n</p>\n<p>')
    _NEW_LINE_REGEX1 = re.compile(r'-\n')
    _STRIP_EMPTY_TAGS=re.compile(r'<[a-z]+?/>')

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
    @print_timing
    def clean_pdf_data(cls, contents):
        """
        Cleans pdf contents extracted by Solr, tries to extract abstract and title.
        :type contents: str
        :return: dict of text, abstract and title
        """
        result_dict = {'text': '', 'abstract': '', 'title': ''}
        contents = contents.translate(cls._CHAR_REPLACEMENT)
        contents = cls._NEW_LINE_REGEX.sub('', contents)
        contents = cls._NEW_LINE_REGEX1.sub('', contents)
        contents = cls._STRIP_EMPTY_TAGS.sub('', contents)
        # otherwise etree can't parse it
        contents = contents.replace('encoding="UTF-8"', '')

        #extract body which is article text
        body = etree.fromstring(contents).find('.//{http://www.w3.org/1999/xhtml}body')
        contents = etree.tostring(body, encoding=unicode)
        contents = strip_tags(contents).strip().split('\n')
        contents = [line for line in contents if line]

        title_start = 0
        title = contents[title_start]
        while not len(title) > 5:
            title_start += 1
            title = contents[title_start].strip()
        result_dict['title'] = title
        del contents[0: title_start + 1]

        result_dict['abstract'], abs_end_index = cls._extract_abstract(contents)

        # remove abstract from rest of the pdf
        contents = contents[abs_end_index:]
        for i in range(len(contents)):
            if len(contents[i].split())<=2:
                contents[i]+='\n'
        result_dict['text'] = ' '.join(contents)
        return result_dict
