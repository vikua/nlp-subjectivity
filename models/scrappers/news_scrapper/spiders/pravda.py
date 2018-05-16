import scrapy

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import cld3


def get_lang(txt):
    return cld3.get_language(txt).language


class PravdaNews(scrapy.Spider):
    name = 'pravda_news'

    def start_requests(self):
        now = datetime.now()
        year_ago = now - relativedelta(years=3)

        delta = now - year_ago

        for i in range(delta.days + 1):
            date = year_ago + timedelta(days=i)
            date_str = date.strftime('%d%m%Y')

            url = 'https://www.pravda.com.ua/news/date_{}/'.format(date_str)
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        articles = response.css('div.block.block_news_all > div.news.news_all > div.article')
        for article in articles:
            href = article.css('a::attr(href)').extract_first()
            yield scrapy.Request(response.urljoin(href), callback=self.parse_article)

    def parse_article(self, response):
        article = response.css('div.post.post_news')

        heading = article.css('h1.post_news__title::text').extract_first()
        date = article.css('div.post_news__date::text')
        tags = article.css('div.post__tags > span.post__tags__item > a::text')

        text = article.css('div.post_news__text').xpath('p//text()')
        body = ' '.join(text.extract())

        if heading and body:
            yield {
                'heading': heading,
                'url': response.request.url,
                'date': date.extract_first(),
                'tags': tags.extract(),
                'body': body,
                'lang': get_lang(body)
            }


class PravdaPublications(scrapy.Spider):
    name = 'pravda_pub'

    num_pages = 300

    def start_requests(self):
        for i in range(1, self.num_pages + 1):
            yield scrapy.Request('https://www.pravda.com.ua/articles/page_{}'.format(i))

    def parse(self, response):
        articles = response.css('div.articles.articles_all > div.article.article_list')
        for article in articles:
            link = article.css('a::attr(href)').extract_first()
            if not link.startswith('http'):
                yield scrapy.Request(response.urljoin(link), callback=self.parse_pravda)
            elif 'epravda' in link:
                yield scrapy.Request(response.urljoin(link), callback=self.parse_epravda)
            elif 'eurointegration' in link:
                link = link.replace('/rus/', '/')
                yield scrapy.Request(response.urljoin(link), callback=self.parse_euro)

    def parse_pravda(self, response):
        article = response.css('div.layout-main div.cols.clearfix')

        heading = article.css('div.article__header > h1::text').extract_first()
        authors = article.css('div.article__header > div.post_news__author a::text')
        date = article.css('div.article__header > div.post_news__date::text')

        text = article.css('div.post.post_news.post_article > div.post_news__text').xpath('p//text()')
        body = ' '.join(text.extract())

        tags = article.css('div.post.post_news.post_article > div.post__tags a::text')

        if heading and body:
            lang = get_lang(body)
            yield {
                'heading': heading,
                'url': response.request.url,
                'authors': authors.extract(),
                'date': date.extract_first(),
                'tags': tags.extract(),
                'body': body,
                'lang': lang
            }

    def parse_epravda(self, response):
        article = response.css('div.layout.layout_second > article')

        heading = article.css('header.post__header > h1::text').extract_first()
        date = article.css('header.post__header > div.post__time::text')
        authors = article.css('header.post__header > div.post__time').xpath('span//text()')

        text = article.css('div.article_content div.post__text').xpath('p//text()')
        body = ' '.join(text.extract())

        tags = article.css('div.post__tags > span > a::text')

        if heading and body:
            lang = get_lang(body)
            yield {
                'heading': heading,
                'url': response.request.url,
                'authors': authors.extract(),
                'date': date.extract_first(),
                'tags': tags.extract(),
                'body': body,
                'lang': lang
            }

    def parse_euro(self, response):
        article = response.css('div.fblock > div.rpad')

        heading = article.css('h1.title::text').extract_first()
        authors = article.css('span.dt2 > b::text')
        date = article.css('span.dt2::text')

        text = article.css('div.text').xpath('p//text()')
        body = ' '.join(text.extract())

        if heading and body:
            lang = get_lang(body)
            yield {
                'heading': heading,
                'url': response.request.url,
                'authors': authors.extract(),
                'date': date.extract_first(),
                'body': body,
                'lang': lang
            }


class PravdaColumn(PravdaPublications):
    name = 'pravda_cols'

    num_pages = 472

    def start_requests(self):
        for i in range(1, self.num_pages + 1):
            yield scrapy.Request('https://www.pravda.com.ua/columns/page_{}'.format(i))

    def parse(self, response):
        cols = response.css('div.columns_all > div.article.article_column')
        for col in cols:
            link = col.css('a::attr(href)').extract_first()
            if not link.startswith('http'):
                yield scrapy.Request(response.urljoin(link), callback=self.parse_pravda)
            elif 'epravda' in link:
                yield scrapy.Request(response.urljoin(link), callback=self.parse_epravda)
            elif 'eurointegration' in link:
                link = link.replace('/rus/', '/')
                yield scrapy.Request(response.urljoin(link), callback=self.parse_euro)
