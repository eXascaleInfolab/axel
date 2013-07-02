from __future__ import division
from collections import defaultdict
import nltk
import re
import itertools
from nltk.metrics.association import BigramAssocMeasures as bigram_assoc

from django.db import models
import MicrosoftNgram

from axel.libs import nlp

ms_ngram_service = MicrosoftNgram.LookupService('37a80cca-9fee-487f-9bbd-c45f252534df',
                                                'bing-body/apr10/5')

THE_EST_COUNT = 0.07  # Rough percentage of the word "the" in the corpus, to estimate real word count


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
        """
        :param text: text to parse, can be more than on sentence.
        """
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
                for i in range(1, 6):
                    for pos_ngram in nltk.ngrams(pos_tag_sent, i):
                        ngram, pos_seq = zip(*[x.rsplit('/', 1) for x in pos_ngram])
                        ngram = u' '.join(ngram)
                        pos_seq = u' '.join(pos_seq)
                        if ngram not in existing:
                            log_prob = ms_ngram_service.GetJointProbability(ngram.encode('utf-8'))
                            print ngram, log_prob, pos_seq
                            Ngram.objects.create(value=ngram, log_prob=log_prob, pos_seq=pos_seq)
                            existing.add(ngram)


class Sentence(models.Model):
    sentence1 = models.TextField()
    sentence2 = models.TextField()

    def __unicode__(self):
        """String representation"""
        return self.sentence1

    @classmethod
    def _tokenize(cls, sentence):
        """Tokenize sentence and return lists of tokens."""
        sentence = re.sub(r'[!.?]', '', sentence)
        tokens = nltk.regexp_tokenize(sentence, nlp.Stemmer.TOKENIZE_REGEXP)
        tokens = [list(x[1]) for x in itertools.groupby(tokens, lambda y: Ngram.PUNKT_RE.match(y))
                  if not x[0]]
        return tokens

    @classmethod
    def _tokenize_positions(cls, sentence):
        """
        Tokenize sentence and return lists of tokens with corresponding positions in the sentence.
        """
        sentence = re.sub(r'[!.?]', '', sentence)
        tokens = nltk.regexp_tokenize(sentence, nlp.Stemmer.TOKENIZE_REGEXP)
        index = 0
        positional_tokens = []
        for token in tokens:
            positional_tokens.append((token, index, index + len(token)))
            index += len(token)

        tokens = [list(x[1]) for x in
                  itertools.groupby(positional_tokens, lambda y: Ngram.PUNKT_RE.match(y[0]))
                  if not x[0]]
        return tokens

    @classmethod
    def get_sentence_scores(cls, sentence):
        """Return averaged probability scores for the sentence using 2- to 5-ngram splits"""
        scores = defaultdict(list)
        for tokens in Sentence._tokenize(sentence):
            for i in range(1, 6):
                for ngram in nltk.ngrams(tokens, i):
                    log_prob = Ngram.objects.get(value=' '.join(ngram)).log_prob
                    scores[i].append(log_prob)
        return dict([(i, sum(probs)/len(probs) if probs else 0) for i, probs in
                     sorted(scores.items(), key=lambda x: x[0])])

    @classmethod
    def get_positional_metrics_data(cls, function='less_than_random_ngrams'):
        """Gets positional data to calculate metrics"""
        positional_data = {}
        for sentence in cls.objects.all():
            positional_data[sentence.id] = [(pos_start, pos_end) for _, pos_start, pos_end in
                                            getattr(sentence, function)()]
        return positional_data

    def less_than_random_ngrams(self):
        """Returns diverging ngrams in the sentence"""
        # TODO: Exclude 100 most frequent words?

        # def _recursive_diverge(tokens, i, parent_prob):
        #     """
        #     Recursively calculate child ngrams normalized log probabilities
        #     (normalization by number of tokens)
        #     """
        #     if i == 2:
        #         yield tokens
        #     else:
        #         ngram1, ngram2 = nltk.ngrams(tokens, i-1)
        #         log_prob = Ngram.objects.get(value=' '.join(ngram1)).log_prob + \
        #                    Ngram.objects.get(value=' '.join(ngram2)).log_prob
        #         if log_prob > parent_prob:
        #             for div_ngram in _recursive_diverge(ngram2, i-1, log_prob):
        #                 yield div_ngram
        #         else:
        #             for div_ngram in _recursive_diverge(ngram1, i-1, log_prob):
        #                 yield div_ngram

        # sentence can contain more than one sequence of tokens
        position_data = []
        for tokens in Sentence._tokenize_positions(self.sentence1):
            for unigram1, unigram2 in nltk.ngrams(tokens, 2):
                bigram = ' '.join([unigram1[0], unigram2[0]])
                log_prob = Ngram.objects.get(value=bigram).log_prob
                sum_log_prob = Ngram.objects.get(value=unigram1[0]).log_prob + \
                               Ngram.objects.get(value=unigram2[0]).log_prob
                if log_prob < sum_log_prob:
                    # report bigram with position
                    position_data.append((bigram, unigram1[1], unigram2[2]))

                # for div_ngram in _recursive_diverge(ngram, 5, log_prob):
                #     divergences.append(' '.join(div_ngram))
        return position_data

    def small_likelihood_ratio(self):
        """
        Tries to identify errors using likelihood ration test under binomial distribution assumption
        """
        # TODO: here
        bigram_assoc.likelihood_ratio()


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
    start_pos_orig = models.IntegerField()
    end_pos_orig = models.IntegerField()
    start_pos_new = models.IntegerField()
    end_pos_new = models.IntegerField()
    sentence = models.ForeignKey(Sentence)
    edit1 = models.CharField(max_length=255)
    # Edit2 is not null when type is REPLACE
    edit2 = models.CharField(max_length=255, null=True)

    @classmethod
    def calculate_positional_metrics(cls, position_data):
        """
        Calculate precision and recall given the positional data for
        possible incorrect places in the data.
        :param position_data: of form {sentence_id: set((start_pos, end_pos), ...), ...}
        :type position_data: dict
        """
        tp = 0
        fp = 0
        fn = 0
        temp_edit_data = cls.objects.values_list('sentence', 'start_pos_orig', 'end_pos_orig')
        orig_edit_data = defaultdict(set)
        for sent_id, start_pos, end_pos in temp_edit_data:
            orig_edit_data[sent_id].add((start_pos, end_pos))

        for sent_id, orig_sent_edit_data in orig_edit_data.iteritems():
            if sent_id in position_data:
                sent_edit_data = sorted(position_data[sent_id])
                print sent_edit_data
                index = 0
                for start_pos, end_pos in sorted(orig_sent_edit_data):
                    if start_pos > sent_edit_data[index][0] and end_pos < sent_edit_data[index][1]:
                        tp += 1
                        index += 1
                    else:
                        fp += 1
            else:
                fn += len(orig_sent_edit_data)
        return tp, fp, fn

    @classmethod
    def calculate_final_metrics(cls, edit_data):
        """Calculate precision and recall given the edit data"""
        orig_edit_data = set(cls.objects.values_list())
        precision = len(edit_data - orig_edit_data) / edit_data
        recall = len(orig_edit_data - edit_data) / edit_data

        return precision, recall
