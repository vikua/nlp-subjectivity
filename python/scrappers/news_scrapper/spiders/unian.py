import scrapy
import cld3


class Unian(scrapy.Spider):
    name = 'unian'

    start_urls = [
        'https://www.unian.ua/war',
        'https://www.unian.ua/politics',
        'https://economics.unian.ua/detail/publications',
    ]

    def parse(self, response):
        pages = response.css('div.pagerfanta > ul.pagination > li > a::text').extract()
        if not pages:
            raise ValueError('No pages found')

        last_page = int(pages[-1])

        self.log('!!! parse pages {}'.format(last_page))

        self.parse_page(response)  # parse first

        for page in range(2, last_page + 1):
            url = '?page={}'.format(page)
            yield scrapy.Request(response.urljoin(url), callback=self.parse_page)

    def parse_page(self, response):
        articles = response.css('section.publications-archive > div.gallery-item > a::attr(href)').extract()
        for href in articles:
            yield scrapy.Request(response.urljoin(href), callback=self.parse_item)

    def parse_item(self, response):
        article = response.css('section.article-column > div.article-text')

        heading = article.css('h1::text')

        date = article.css('div.item.time::text')

        sub_heading = article.css('div.like-h2::text').extract_first()
        if not sub_heading:
            sub_heading = article.css('h2::text').extract_first()

        text = article.css('div').xpath('p//text()').extract()

        if sub_heading:
            body = [sub_heading] + text
        else:
            body = text

        body = ' '.join(body)
        lang = cld3.get_language(body).language

        tags = article.css('div.tags > a::text')

        yield {
            'heading': heading.extract_first(),
            'date': date.extract_first(),
            'tags': tags.extract(),
            'lang': lang,
            'body': body
        }