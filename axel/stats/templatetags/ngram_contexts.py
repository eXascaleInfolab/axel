from django.utils.safestring import mark_safe
from django import template

from templatetag_sugar import register
from templatetag_sugar.parser import Variable
from templatetag_sugar.register import tag

register = template.Library()


@tag(register, [Variable('ngram'), Variable('ngram_context')])
def highlight_ngram_context(context, ngram, ngram_context):
    """Highlight ngram in the context"""
    return mark_safe(ngram_context.lower().replace(ngram, u'<span class="error">{0}</span>'.format(ngram)))
