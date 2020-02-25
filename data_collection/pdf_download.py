import scrapy
from scrapy.http import Request
from scrapy.crawler import CrawlerProcess
import logging

class PsySpider(scrapy.Spider):
    name = "psy"
    start_urls = ["https://www.nite.org.il/psychometric-entrance-test/preparation/hebrew-practice-tests/"]

    def parse(self, response):
        # for href in response.css('a[href$=".pdf" and target="_blank"]::attr(href)').extract():
        for href in response.xpath('//h4/a[contains(@href,".pdf")]/@href').extract():
            if 'hebrew' not in href:
                yield Request(
                    url=response.urljoin(href),
                    callback=self.save_pdf
                )

    def save_pdf(self, response):
        path = response.url.split('/')[-1]
        self.logger.info('Saving PDF %s', path)
        with open(rf'tests/{path}', 'wb') as f:
            f.write(response.body)


def crawl_psy():
    process = CrawlerProcess()
    process.crawl(PsySpider)
    process.start()


if __name__ == '__main__':
    crawl_psy()
