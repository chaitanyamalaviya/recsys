import json

from scrapy.http import Request
from scrapy.spider import BaseSpider


class MySpider(BaseSpider):
    name = 'myspider'
    start_urls = (
        # Add here more urls. Alternatively, make the start urls dynamic
        # reading them from a file, db or an external url.
        'https://www.facebook.com/TiltedKiltEsplanade',
    )

    graph_url = 'https://graph.facebook.com/{name}'
    feed_url = 'https://www.facebook.com/feeds/page.php?id={id}&format=rss20'

    def start_requests(self):
        for url in self.start_urls:
            # This assumes there is no trailing slash
            name = url.rpartition('/')[2]
            yield Request(self.graph_url.format(name=name), self.parse_graph)

    def parse_graph(self, response):
        data = json.loads(response.body)
        return Request(self.feed_url.format(id=data['id']), self.parse_feed)

    def parse_feed(self, response):
        # You can use the xml spider, xml selector or the feedparser module
        # to extract information from the feed.
        self.log('Got feed: %s' % response.body[:100])
