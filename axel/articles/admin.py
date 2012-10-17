from django.contrib import admin
from axel.articles.models import Article, ArticleAuthor, Author, Venue


admin.site.register(Venue)
admin.site.register(ArticleAuthor)
admin.site.register(Article)
admin.site.register(Author)
