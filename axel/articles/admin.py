from django.contrib import admin
from axel.articles.models import Article, ArticleAuthor, Author, Venue


class ArticleAdmin(admin.ModelAdmin):
    fields = ('venue', 'year', 'link', 'citations', 'pdf')
    list_display = ('title', 'venue', 'year')

admin.site.register(Venue)
admin.site.register(ArticleAuthor)
admin.site.register(Article, ArticleAdmin)
admin.site.register(Author)
