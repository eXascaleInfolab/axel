from collections import defaultdict
import nltk
import re
import itertools
import math

from django.db import models
import MicrosoftNgram

from axel.libs import nlp

ms_ngram_service = MicrosoftNgram.LookupService('37a80cca-9fee-487f-9bbd-c45f252534df',
                                                'bing-body/apr10/5')


class Ngram(models.Model):
    """Describes ngram"""
    value = models.TextField()
    log_prob = models.FloatField()
    pos_seq = models.CharField(max_length=255)

    PUNKT_RE = re.compile(r'[`~/%\*\+\[\]\.?!,":;()\'|]+')

    class Meta:
        """Meta info"""
        ordering = ['value']

    def __unicode__(self):
        """String representation"""
        return self.value

    @classmethod
    def create_from_text(cls, text):
        existing = set(Ngram.objects.values_list("value", flat=True))
        for sentence in re.split(r'[.?!]', text):
            sentence = sentence.strip()
            if not sentence:
                continue
            pos_tag_sents = nltk.pos_tag(nltk.regexp_tokenize(sentence, nlp.Stemmer.TOKENIZE_REGEXP))
            # join ngrams with tags
            pos_tag_sents = ['/'.join(x) for x in pos_tag_sents]
            pos_tag_sents = [list(x[1]) for x in itertools.groupby(pos_tag_sents,
                                                lambda x: cls.PUNKT_RE.match(x)) if not x[0]]
            for pos_tag_sent in pos_tag_sents:
                for i in [1]:
                    for pos_ngram in nltk.ngrams(pos_tag_sent, i):
                        ngram, pos_seq = zip(*[x.rsplit('/', 1) for x in pos_ngram])
                        ngram = u' '.join(ngram)
                        pos_seq = u' '.join(pos_seq)
                        if ngram not in existing:
                            log_prob = ms_ngram_service.GetJointProbability(ngram.encode('utf-8'))
                            print ngram, log_prob, pos_seq
                            Ngram.objects.create(value=ngram, log_prob=log_prob, pos_seq=pos_seq)


class Sentence(models.Model):
    sentence1 = models.TextField()
    sentence2 = models.TextField()

    def __unicode__(self):
        """String representation"""
        return self.sentence1

    @classmethod
    def tokenize(cls, sentence):
        """Tokenize sentence and return lists of tokens"""
        tokens = nltk.regexp_tokenize(sentence, nlp.Stemmer.TOKENIZE_REGEXP)
        tokens = [list(x[1]) for x in itertools.groupby(tokens,
                                                    lambda x: Ngram.PUNKT_RE.match(x)) if not x[0]]
        return tokens

    @classmethod
    def get_sentence_scores(cls, sentence):
        """Return averaged probability scores for the sentence using 2- to 5-ngram splits"""
        scores = defaultdict(list)
        for tokens in Sentence.tokenize(sentence):
            for i in range(2, 6):
                for ngram in nltk.ngrams(tokens, i):
                    log_prob = Ngram.objects.get(value=' '.join(ngram)).log_prob
                    scores[i].append(log_prob)
        return dict([(i, sum(probs)/len(probs) if probs else 0) for i, probs in
                     sorted(scores.items(), key=lambda x: x[0])])

    def divergence_iter(self):
        """Yields diverging ngrams in the sentence"""

        def _recursive_diverge(tokens, i, parent_prob):
            """
            Recursively calculate child ngrams normalized log probabilities
            (normalization by number of tokens)
            """
            if i == 2:
                yield tokens
            else:
                ngram1, ngram2 = nltk.ngrams(tokens, i-1)
                log_prob = Ngram.objects.get(value=' '.join(ngram1)).log_prob + \
                           Ngram.objects.get(value=' '.join(ngram2)).log_prob
                if log_prob > parent_prob:
                    for div_ngram in _recursive_diverge(ngram2, i-1, log_prob):
                        yield div_ngram
                else:
                    for div_ngram in _recursive_diverge(ngram1, i-1, log_prob):
                        yield div_ngram

        # sentence can contain more than one sequence of tokens
        divergences = []
        for tokens in Sentence.tokenize(self.sentence1):
            for bigram in nltk.ngrams(tokens, 2):
                log_prob = Ngram.objects.get(value=' '.join(bigram)).log_prob
                sum_log_prob = Ngram.objects.get(value=bigram[0]).log_prob + \
                               Ngram.objects.get(value=bigram[1]).log_prob
                if log_prob < sum_log_prob:
                    # report bigram
                    divergences.append(' '.join(bigram))

                # for div_ngram in _recursive_diverge(ngram, 5, log_prob):
                #     divergences.append(' '.join(div_ngram))
        return divergences


class Edit(models.Model):
    DELETE = 'DEL'
    INSERT = 'INS'
    REPLACE = 'REP'

    EDIT_TYPES = (
        (DELETE, 'delete'),
        (INSERT, 'insert'),
        (REPLACE, 'replace'),
    )
    edit_type = models.CharField(max_length=3, choices=EDIT_TYPES)
    sentence = models.ForeignKey(Sentence)
    edit1 = models.CharField(max_length=255)
    # Edit2 is not null when type is REPLACE
    edit2 = models.CharField(max_length=255, null=True)
