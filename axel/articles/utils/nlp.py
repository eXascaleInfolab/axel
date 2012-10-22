"""Utils to do text processing with NLTK"""
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder


def lemmatize(text):
    """Simple lemmatization"""
    lmtzr = WordNetLemmatizer()
    # split on whitespace
    result = []
    for word in text.split():
        result.append(lmtzr.lemmatize(word))
    return ' '.join(result)

_PUNKT_RE = re.compile(r'[`~/%\*\+\[\]\-.?!,":;()\'|0-9]')

_STOPWORDS = ['per', 'could', 'like', 'better', 'community', 'within', 'via',
              'around', 'seen', 'would', 'along', 'successful', 'may', 'without',
              'including', 'given', 'today', 'yield', 'towards']
_STOPWORDS.extend(nltk.corpus.stopwords.words('english'))


def collocations(text):
    """Extract collocations from text"""

    def filter_punkt(word):
        return _PUNKT_RE.match(word)

    tokens = nltk.wordpunct_tokenize(text)
    text = nltk.Text(tokens)
    finder = BigramCollocationFinder.from_words(text)
    # remove collocation from 2 equal words
    finder.apply_ngram_filter(lambda x, y: x == y)
    # remove punctuation
    finder.apply_word_filter(filter_punkt)
    finder.apply_freq_filter(3)
    finder.apply_word_filter(lambda w: w in _STOPWORDS)
    bigram_measures = BigramAssocMeasures()
    collocs = finder.score_ngrams(bigram_measures.raw_freq)
    return collocs
