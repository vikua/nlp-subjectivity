import re

import scrapy
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import cld3


class SegodnyaPolitics(scrapy.Spider):
    name = 'segodnya_pol'

    def base_url(self):
        return 'https://ukr.segodnya.ua/politics/archive'

    def start_requests(self):
        now = datetime.now()
        years_ago = now - relativedelta(years=3)

        delta = now - years_ago

        for i in range(delta.days + 1):
            date = years_ago + timedelta(days=i)
            date_str = date.strftime('%d-%m-%Y')
            url = '{}/{}.html'.format(self.base_url(), date_str)
            yield scrapy.Request(url, callback=self.parse, headers={'Accept': '*/*',
                                                                    'Host': 'ukr.segodnya.ua'})

    def parse(self, response):
        news = response.css('div.content-blocks > div.news-block-wrapper.news-block')
        for n in news:
            href = n.css('a::attr(href)').extract_first()
            yield scrapy.Request(response.urljoin(href), callback=self.parse_news)

    def parse_news(self, response):
        self.log(response.body)
        article = response.css('div.article-content')

        heading = article.css('div.title > h1::text')
        date = article.css('div.title > div > span::text')

        text = article.css('span._ga1_on_').xpath('p//text()')
        body = ' '.join(text.extract())
        lang = cld3.get_language(body).language

        tags = response.css('div.article-content > div.tag > a::text')

        yield {
            'heading': heading.extract_first(),
            'date': date.extract_first(),
            'lang': lang,
            'tags': tags.extract(),
            'body': body
        }

class SegodnyaEconom(SegodnyaPolitics):
    name = 'segodnya_econ'

    def base_url(self):
        return 'https://ukr.segodnya.ua/economics/archive'