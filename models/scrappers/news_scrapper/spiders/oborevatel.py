import scrapy
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import cld3


class ObozrevatelEconom(scrapy.Spider):
    name = 'obo_econ'

    url_templates = [
        'https://www.obozrevatel.com/ukr/economics/news-of-the-day/{}.htm',
        'https://www.obozrevatel.com/ukr/news-of-the-day/{}.htm'
    ]

    def start_requests(self):
        now = datetime.now()
        years_ago = now - relativedelta(years=3)

        delta = now - years_ago

        for i in range(delta.days + 1):
            date = years_ago + timedelta(days=i)
            date_str = date.strftime('%d-%m-%Y')
            for url in self.url_templates:
                yield scrapy.Request(url.format(date_str), callback=self.parse)

    def parse(self, response):
        articles = response.css('div.section-news-title-img-text article.news-title-img-text')
        for article in articles:
            link = article.css('a::attr(href)').extract_first()
            yield scrapy.Request(response.urljoin(link), self.parse_article)

    def parse_article(self, response):
        article = response.css('main.main-col > div.main-col__left')

        heading = article.css('div.news-full__head > h1.news-full__title::text')
        date = article.css('div.news-full__head > time.news-full__date::text')

        text = article.css('div.news-full__text').xpath('p//text()')
        tags = article.css('div.news-full-tags > div > span::text')

        body = ' '.join(text.extract())
        lang = cld3.get_language(body).language

        yield {
            'heading': heading.extract_first(),
            'date': date.extract_first(),
            'lang': lang,
            'tags': tags.extract(),
            'url': response.request.url,
            'body': body
        }

