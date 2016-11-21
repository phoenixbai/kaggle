import scrapy

class PiaoSpider(scrapy.Spider):
    name = "piao"

    def start_requests(self):
        urls = ['http://www.mgtv.com/v/2016/jyj2016/toupiao/']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = "piao.txt"
        print response.body