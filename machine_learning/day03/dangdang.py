# -*- coding: utf-8 -*-
import requests
from lxml import html
import pandas as pd


def spider(isbn, book_list = []):
    url = 'http://search.dangdang.com/?key={}&act=input'.format(isbn)
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"}
    response = requests.get(url, headers = headers)
    html_data = response.text
    selector = html.fromstring(html_data)
    url_list = selector.xpath('//div[@id="search_nature_rg"]/ul/li')
    for li in url_list:
        book_name = li.xpath('p[@class="name"]/a/text()')[0]
        print(book_name)
        img_src = li.xpath('a[1]/img/@src')[0]
        print(img_src)
        price = li.xpath('p[@class="price"]/span[1]/text()')
        price = "¥0" if len(price) == 0 else price[0]
        price = float(price.replace("¥", ""))
        book_list.append({
            
            "book_name": book_name,
            "img_src": img_src,
            "price": price
            })
    
    for book in book_list:
        print(book)
        
    df = pd.DataFrame(book_list)
    df.to_csv("dangdang.csv", index = False)
    
if __name__ == '__main__':
    str = '9787115546081'
    spider(str)