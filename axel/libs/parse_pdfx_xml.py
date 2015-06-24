#! /usr/bin/python
# -*- coding: UTF-8 -*-

""" Script to Parse PDFX XML for input to plain text just of specific regions.
    Will output logical region markers e.g. "::TITLE::" for each
    of the 10 regions considered: title, abstact, h1, h2, h3,
    introduction, conclusion, normal body text, caption, bibliography."""
import re

__author__ = 'Alex Constantin, Roman Prokofyev'
__version__ = '1.0'
__date__ = 'August 2013'
__contact__ = 'aconstantin@cs.man.ac.uk'

import os
import sys
from lxml import etree
from itertools import chain

xml_file = None
uid = 0
option_dict = {}

precs = {}
recalls = {}
f1s = {}


def get(elem, tag):
    """ Get the text of each <tag> from the etree element, sanitised """

    res = []
    for tag_elem in elem.xpath(tag):
        prepared_str = prepare(stringify_children(tag_elem, itrp_win=option_dict.get('itrp_win')))
        if prepared_str:
            res.append(prepared_str)
    return res


def prepare(str):
    # lower() removed
    lines = str.split("\n")

    for i, l in enumerate(lines):
        lines[i] = " ".join(lines[i].strip().split())

    return " . \n".join([l for l in lines if len(l.strip())])


#   return " . \n".join(lines)


def stringify_children(node, itrp_win, tails=False):
    """ Option to exclude the element tail is only allowed for the outermost element """

    if node is None:
        return ''

    txt = node.text
    if not txt:
        txt = ''

    if tails:
        tl = node.tail
        if not tl:
            tl = ''
        parts = ([txt, ' '] +
                 list(chain(*(stringify_children(c, itrp_win, tails=True) for c in list(node)))) +
                 [' ', tl])
    else:
        parts = ([txt, ' '] +
                 list(chain(*(stringify_children(c, itrp_win, tails=True) for c in list(node)))))

    # remove possible Nones in texts and tail
    return ''.join([p for p in parts if p])


#===============================================================================
# PARSE PDFX XML
#===============================================================================

def parse_pdfx_xml(xml_file_path):
    """ Get the relevant bits from the PDFX XML
        (title, abstract, h1, h2, h3, introduction, conclusion,
         other body text, captions and bibliogrpahy)
    """

    global option_dict
    global xml_file
    global precs
    global recalls
    global f1s

    xml_file = xml_file_path

    if os.path.isfile(xml_file):
        parser = etree.XMLParser(recover=True, ns_clean=True)
        pdfx_tree = etree.parse(xml_file, parser)
        pdfx_root = pdfx_tree.getroot()

        out = u''

        #===================================================================
        # TITLE
        #===================================================================
        pdfx_titles = get(pdfx_root, "./article//title-group/article-title")
        #      print "\npdfx_titles:", pdfx_titles

        if len(pdfx_titles):
            out += u"::TITLE::\n\n"
            for title in pdfx_titles:
                out += title + u".\n\n"

        #===================================================================
        # ABSTRACT
        #===================================================================
        pdfx_abstracts = get(pdfx_root, ".//abstract")
        #      print "\npdfx_abstracts:", pdfx_abstracts

        if len(pdfx_abstracts):
            out += u"::ABSTRACT::\n\n"
            for abs in pdfx_abstracts:
                out += abs + u".\n\n"

        #===================================================================
        # H1
        #===================================================================
        pdfx_h1s = get(pdfx_root, "./article/body/section/h1")

        blacklist = ["abbreviations", "acknowledgments", "acknowledgment", "acknowledgement",
                     "acknowledgements",
                     "additional material", 'additional files', "author contributions",
                     "author's contributions", "authors' contributions",
                     'author details', "author summary", "competing interests",
                     "pre-publication history", "references",
                     "structure reports online", "supplementary data", "supplementary material"]
        pdfx_h1s = [h1 for h1 in pdfx_h1s if h1 not in blacklist]
        #      print "\npdfx_h1s (filtered):", pdfx_h1s

        if len(pdfx_h1s):
            out += u"::H1::\n\n"
            for h1 in pdfx_h1s:
                out += h1 + ".\n\n"

        #===================================================================
        # H2
        #===================================================================
        pdfx_h2s = get(pdfx_root, "./article/body//section/h2")
        #      print "\npdfx_h2s:", pdfx_h2s

        if len(pdfx_h2s):
            out += u"::H2::\n\n"
            for h2 in pdfx_h2s:
                out += h2 + u".\n\n"

        #===================================================================
        # H3
        #===================================================================
        pdfx_h3s = get(pdfx_root, "./article/body//section/h3")
        #      print "\npdfx_h3s:", pdfx_h3s

        if len(pdfx_h3s):
            out += u"::H3::\n\n"
            for h3 in pdfx_h3s:
                out += h3 + u".\n\n"

        #===================================================================
        # INTRODUCTION and CONCLUSION BODIES
        #===================================================================
        intro_flag = False
        backgr_flag = False
        pdfx_intro = get(pdfx_root,
                         ".//section[@class='deo:Introduction']//region[@class='DoCO:TextChunk']")
        # Fallback to Background if no conclusion is present
        if not len(pdfx_intro):
            pdfx_intro = get(pdfx_root,
                             ".//section[@class='deo:Background']//region[@class='DoCO:TextChunk']")
            backgr_flag = True
        else:
            intro_flag = True
        #      print "\npdfx_intro:", pdfx_intro

        if len(pdfx_intro):
            out += u"::INTRO::\n\n"
            for intro in pdfx_intro:
                out += intro + u".\n\n"

        concl_flag = False
        discussion_flag = False
        pdfx_concl = get(pdfx_root,
                         ".//section[@class='deo:Conclusion']//region[@class='DoCO:TextChunk']")
        # Fallback to Discussion if no conclusion is present
        if not len(pdfx_concl):
            pdfx_concl = get(pdfx_root,
                             ".//section[@class='deo:Discussion']//region[@class='DoCO:TextChunk']")
            discussion_flag = True
        else:
            concl_flag = True
        #      print "\npdfx_concl:", pdfx_concl

        if len(pdfx_concl):
            out += u"::CONCLUSION::\n\n"
            for concl in pdfx_concl:
                out += concl + u".\n\n"

        #===================================================================
        # OTHER BODIES (possibles included)
        #===================================================================
        pdfx_bodies = []
        if intro_flag and concl_flag:
            pdfx_bodies = get(pdfx_root,
                              ".//section[not(@class='deo:Introduction') and not(@class='deo:Conclusion')]//region[@class='DoCO:TextChunk']")
        elif backgr_flag and concl_flag:
            pdfx_bodies = get(pdfx_root,
                              ".//section[not(@class='deo:Background') and not(@class='deo:Conclusion')]//region[@class='DoCO:TextChunk']")
        elif intro_flag and discussion_flag:
            pdfx_bodies = get(pdfx_root,
                              ".//section[not(@class='deo:Introduction') and not(@class='deo:Discussion')]//region[@class='DoCO:TextChunk']")
        elif backgr_flag and discussion_flag:
            pdfx_bodies = get(pdfx_root,
                              ".//section[not(@class='deo:Background') and not(@class='deo:Discussion')]//region[@class='DoCO:TextChunk']")

        #      print "\npdfx_bodies:", pdfx_bodies

        if len(pdfx_bodies):
            out += u"::BODY::\n\n"
            for body in pdfx_bodies:
                if body not in pdfx_concl + pdfx_intro:
                    #               print ">", [body]
                    out += body + u".\n\n"

        #===================================================================
        # FIG/TABLE CAPTIONS
        #===================================================================
        pdfx_captions = [] #get(pdfx_root, ".//caption")
        #      print "\npdfx_captions:", pdfx_captions

        if len(pdfx_captions):
            out += u"::CAPTION::\n\n"
            for cap in pdfx_captions:
                out += cap + u".\n\n"

        #=========================================================================
        # BIB ITEMS
        #=========================================================================
        pdfx_bib_items = [] #get(pdfx_root, ".//ref-list/ref")
        #      #print "\npdfx_bib_items:", pdfx_bib_items

        if len(pdfx_bib_items):
            out += u"::BIBLIOGRAPHY::\n\n"
            for bib in pdfx_bib_items:
                out += bib + u" .\n\n"

        # removing hyphenation
        out = re.sub(r'\b- \b', '', out)
        return out

    else:
        print "** ERROR ** Cannot locate XML file:", xml_file
        return None


if __name__ == "__main__":
    from optparse import OptionParser

    usage = '%prog [-diff] <pdfx_xml>'
    description = "Parser for PDFX XML to output plain text with logical region markers e.g. '::TITLE::' " + \
                  "to pass to KPEX as abridged input."
    parser = OptionParser(usage=usage, description=description)

    parser.add_option('-w', '--itrp-win', action="store", type="int", dest="itrp_win",
                      default=0, help="ITRP sentence window to ignore (NOT IMPLEMENTED YET).")

    options, args = parser.parse_args()
    option_dict = vars(options)

    if len(args) != 1:
        print "Usage: ./parse_pdfx_xml.py <pdfx_xml>"
        sys.exit(1)

    sys.exit(parse_pdfx_xml(args))
