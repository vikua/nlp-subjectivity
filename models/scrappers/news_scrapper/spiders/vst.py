import scrapy
import logging


class VestiPublications(scrapy.Spider):
    name = 'vesti_pub'
    start_urls = [
        'https://vesti-ukr.com/publications'
    ]

    def pagination_template(self):
        return 'publications/p{}'

    def parse(self, response):
        pages = response.css('div.publications > div.pagination > div.pages a::text').extract()
        if not pages:
            raise ValueError('No pages found')
        last_page = int(pages[-1])

        for page in range(1, last_page + 1):
            url = self.pagination_template().format(page)
            yield scrapy.Request(response.urljoin(url), callback=self.parse_page)

    def parse_page(self, response):
        self.log('Parsing page {}'.format(response.request.url))
        news = response.css('div.publications > div.news > div.new-item')
        for n in news:
            url = n.css('a::attr(href)').extract_first()
            yield scrapy.Request(response.urljoin(url), callback=self.parse_article)

    def parse_article(self, response):
        article = response.css('div.main.article.article-text')

        heading = article.css('h1::text')
        authors = article.css('div.post-info span.article-authors > a::text')
        date = article.css('div.post-info span.date::text')

        meta = article.css('div.photo > div.new-meta > span::text')
        tags = article.css('div.tags > a > span::text')

        text = article.css('div.article-content').xpath('p//text()')

        yield {
            'heading': heading.extract_first(),
            'metadata': meta.extract_first(),
            'url': response.request.url,
            'authors': authors.extract(),
            'date': date.extract_first(),
            'tags': tags.extract(),
            'body': ' '.join(text.extract())
        }


class VestiInvestigations(VestiPublications):
    name = 'vesti_inv'
    start_urls = [
        'https://vesti-ukr.com/investigations'
    ]

    def pagination_template(self):
        return 'investigations/p{}'


class VestiNews(VestiPublications):
    name = 'vesti_news'
    start_urls = [
        'https://vesti-ukr.com/feed/1-vse-novosti'
    ]

    def pagination_template(self):
        return '1-vse-novosti/{}'

    def parse(self, response):
        pages = response.css('div.main > div.news-list div.pages a::text').extract()
        if not pages:
            raise ValueError('No pages found')
        last_page = int(pages[-1])

        for page in range(1, last_page + 1):
            url = self.pagination_template().format(page)
            yield scrapy.Request(response.urljoin(url), callback=self.parse_page)

    def parse_page(self, response):
        news_links = response.css('div.main > div.news-list > ul > li > a::attr(href)').extract()
        for link in news_links:
            yield scrapy.Request(response.urljoin(link), self.parse_article)

class VestiAttacks(VestiNews):
    name = 'vesti_attacks'
    start_urls = [
            'https://vesti-ukr.com/obyski-v-vestjakh'
    ]

    def pagination_template(self):
        return 'obyski-v-vestjakh/{}'