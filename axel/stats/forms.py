from django import forms


class ScoreCacheResetForm(forms.Form):
    attr = forms.ChoiceField(choices=())

    def __init__(self, model_class, *args, **kwargs):
        super(ScoreCacheResetForm, self).__init__(*args, **kwargs)
        self.fields['attr'].choices = [(score, score) for score in model_class.CACHED_FIELDS]


class NgramBindingForm(forms.Form):
    scoring_function = forms.ChoiceField(choices=())
    pos_tag = forms.CharField(required=False)

    def __init__(self, *args, **kwargs):
        super(NgramBindingForm, self).__init__(*args, **kwargs)
        from axel.stats.views import NgramWordBindingDistributionView
        self.fields['scoring_function'].choices = [(score, score) for score in NgramWordBindingDistributionView.scores()]
