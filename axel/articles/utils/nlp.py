"""Utils to do text processing with NLTK"""
from collections import defaultdict
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import AbstractCollocationFinder
from nltk.stem.porter import PorterStemmer
from django.template import loader, Context

from axel.articles.utils.pdfcleaner import PDFCleaner
from axel.libs.utils import print_timing

class Stemmer:
    """Collection of stemmers"""

    @classmethod
    def stem_wordnet(cls, text):
        """WordNet lemmatizer"""
        lmtzr = WordNetLemmatizer()
        # split on whitespace
        result = []
        for word in nltk.wordpunct_tokenize(text):
            if word.istitle():
                word = word.lower()
            result.append(lmtzr.lemmatize(word))
        return ' '.join(result)


    @classmethod
    def stem_porter(cls, text):
        """Porter stemmer"""
        stemmer = PorterStemmer()
        result = []
        for word in nltk.wordpunct_tokenize(text):
            if word.istitle():
                word = word.lower()
            result.append(stemmer.stem(word))
        return ' '.join(result)

    @classmethod
    def get_method_names(cls):
        """
        :returns: list of tuples containing stem method name with descriptions
        """
        def is_stem_method(attrname, klass=cls, prefix='stem_'):
            return attrname.startswith(prefix) and hasattr(getattr(klass, attrname), '__call__')
        f_names = filter(is_stem_method, dir(cls))
        f_names = [(f_name, getattr(cls, f_name).__doc__) for f_name in f_names]
        return f_names


_PUNKT_RE = re.compile(r'[`~/%\*\+\[\]\-.?!,":;()\'|]+')

_STOPWORDS = {'per', 'could', 'like', 'better', 'community', 'within', 'via', 'around', 'seen',
              'would', 'along', 'successful', 'may', 'without', 'including', 'given', 'today',
              'yield', 'towards', 'whether'}
_STOPWORDS.update(nltk.corpus.stopwords.words('english'))


@print_timing
def collocations(index):
    """
    Extract collocations from n-gram index
    :type index: dict
    :rtype list
    """

    def filter_punkt(word):
        return _PUNKT_RE.match(word)

    def filter_len(word):
        return len(word) < 3

    # do filtration by frequency > 2
    bigram_index = dict([(tuple(k.split()), v) for k, v in index.iteritems()
                         if len(k.split()) == 2 and v > 2])

    # Get abstract finder because we already have index
    finder = AbstractCollocationFinder(None, bigram_index)
    # remove collocation from 2 equal words
    finder.apply_ngram_filter(lambda x, y: x == y)
    # remove punctuation, len and stopwords
    finder.apply_word_filter(filter_punkt)
    finder.apply_word_filter(filter_len)
    finder.apply_word_filter(lambda w: w in _STOPWORDS)

    filtered_collocs = finder.ngram_fd

    # generate possible n-grams
    filtered_collocs = _generate_possible_ngrams(filtered_collocs.items(), index)

    # join tuples
    filtered_collocs = [(score, (' '.join(colloc))) for colloc, score in filtered_collocs]

    return filtered_collocs


@print_timing
def _generate_possible_ngrams(collocs, index):
    """
    Recursively generate all possible n-grams from list of bigrams
    Score is needed because we want to know if n-1 gram is present in the text itself
    or it is always part of n-gram.
    :param collocs: list of bigrams with scores
    :type collocs: list
    :type index: dict
    """
    collocs = collocs[:]
    possible_ngrams = []
    total_len = len(collocs)
    for i in range(total_len):
        bigram_i, score_i = collocs[i]
        if not score_i:
            continue
        for j in range(i+1, total_len):
            bigram_j, score_j = collocs[j]
            # check score_j and score_i is ok
            if not score_j:
                continue

            inter = set(bigram_i).intersection(bigram_j)
            inter_len = len(inter)
            if not inter:
                continue

            # check intersection does not fully contains one of the n-grams
            if inter_len == min((len(bigram_i), len(bigram_j))):
                continue

            bigram_s, bigram_e = None, None
            if set(bigram_i[:inter_len])==inter and set(bigram_j[-inter_len:])==inter:
                bigram_s, bigram_e = bigram_i, bigram_j
            elif set(bigram_j[:inter_len])==inter and set(bigram_i[-inter_len:])==inter:
                bigram_s, bigram_e = bigram_j, bigram_i

            if bigram_s and bigram_e:
                new_ngram = bigram_e+bigram_s[inter_len:]
                # Check new colocation actually present in text
                if ' '.join(new_ngram) in index:
                    min_score = index[' '.join(new_ngram)]
                    possible_ngrams.append((new_ngram, min_score))
                    collocs[i] = (bigram_i, score_i-min_score)
                    collocs[j] = (bigram_j, score_j-min_score)

    # remove zero-score old collocations
    collocs = [(bigram, score) for bigram, score in collocs if score != 0]
    possible_ngrams.extend(collocs)
    if len(possible_ngrams) == len(collocs):
        return possible_ngrams
    else:
        return _generate_possible_ngrams(possible_ngrams, index)


@print_timing
def stem_text(text, stem_func=Stemmer.stem_wordnet):
    """
    Stems text passed from text argument
    :type text: str
    :param stem_func: function that performs word stemming
    :returns: dict containing the results, in form {'title':..., 'abstract':..., 'text':...}
    """
    import traceback
    result = PDFCleaner.clean_pdf_data(text)
    t = loader.select_template(('search/indexes/articles/article_text.txt', ))
    full_text = t.render(Context({'extracted': result}))
    full_text = stem_func(full_text)
    # override full text
    result['text'] = full_text
    return result


@print_timing
def _split_sentences(text):
    """
    Splits text on sentences (collocated-parts) based on re
    :type text: unicode
    :rtype: list
    """
    return re.split(_PUNKT_RE, text)


@print_timing
def build_ngram_index(text):
    """
    Build n-grams from text up to max len in the db, *with actual counts*
    :type text: unicode
    :rtype: defaultdict
    """

    # First step - sentence split.
    text = _split_sentences(text)

    # Second step - ngram generation
    all_ngrams = defaultdict(lambda: 0)
    # TODO: add real value from db
    max_db_size = 5
    for n in range(2, max_db_size+1):
        for sentence in text:
            ngrams = nltk.ngrams(sentence.split(), n)
            for ngram in ngrams:
                all_ngrams[(' '.join(ngram))] += 1
    return all_ngrams
