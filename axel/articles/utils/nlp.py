"""Utils to do text processing with NLTK"""
from collections import defaultdict
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

_PUNKT_RE = re.compile(r'[`~/%\*\+\[\]\-.?!,":;()\'|0-9]+')

_STOPWORDS = ['per', 'could', 'like', 'better', 'community', 'within', 'via',
              'around', 'seen', 'would', 'along', 'successful', 'may', 'without',
              'including', 'given', 'today', 'yield', 'towards']
_STOPWORDS.extend(nltk.corpus.stopwords.words('english'))


def collocations(text):
    """Extract collocations from text"""

    def filter_punkt(word):
        return _PUNKT_RE.match(word)

    def filter_len(word):
        return len(word) < 3

    tokens = nltk.wordpunct_tokenize(text)
    text = nltk.Text(tokens)
    finder = BigramCollocationFinder.from_words(text)
    # remove collocation from 2 equal words
    finder.apply_ngram_filter(lambda x, y: x == y)
    # remove punctuation
    finder.apply_word_filter(filter_punkt)
    finder.apply_word_filter(filter_len)
    # Weird freq filter does not work properly :(
    finder.apply_freq_filter(3)

    finder.apply_word_filter(lambda w: w in _STOPWORDS)
    bigram_measures = BigramAssocMeasures()
    collocs = finder.score_ngrams(bigram_measures.raw_freq)

    # Now do real filtration by frequency
    word_d_n = finder.word_fd.N()
    filtered_collocs = [(int(score*word_d_n), name) for name, score in collocs
                        if int(score*word_d_n) > 2]

    filtered_collocs = _generate_possible_ngrams(filtered_collocs)

    # join tuples
    filtered_collocs = [(score, (' '.join(coloc))) for score, coloc in filtered_collocs]

    return filtered_collocs


def _generate_possible_ngrams(collocs):
    """
    Recursively generate all possible n-grams from list of bigrams
    :param collocs: list of bigrams
    :type collocs: list
    """
    collocs = collocs[:]
    possible_ngrams = []
    total_len = len(collocs)
    for i in range(total_len):
        score_i, bigram_i = collocs[i]
        if not score_i:
            continue
        for j in range(i+1, total_len):
            score_j, bigram_j = collocs[j]
            # check score_j and score_i is ok
            if not score_j:
                continue

            inter = set(bigram_i).intersection(bigram_j)
            if not inter:
                continue
            inter_len = len(inter)

            bigram_s, bigram_e = None, None
            if set(bigram_i[:inter_len])==inter and set(bigram_j[-inter_len:])==inter:
                bigram_s, bigram_e = bigram_i, bigram_j
            elif set(bigram_j[:inter_len])==inter and set(bigram_i[-inter_len:])==inter:
                bigram_s, bigram_e = bigram_j, bigram_i

            if bigram_s and bigram_e:
                min_score = min((score_i, score_j))
                possible_ngrams.append((min_score, bigram_e+bigram_s[inter_len:]))
                collocs[i] = (score_i-min_score, bigram_i)
                collocs[j] = (score_j-min_score, bigram_j)

    # remove zero-score old collocations
    collocs = [(score, bigram) for score, bigram in collocs if score != 0]
    possible_ngrams.extend(collocs)
    if len(possible_ngrams) == len(collocs):
        return possible_ngrams
    else:
        return _generate_possible_ngrams(possible_ngrams)
