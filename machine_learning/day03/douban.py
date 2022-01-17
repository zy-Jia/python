# -*- coding: utf-8 -*-
import requests
from lxml import html
import pandas as pd
import json


def spider():
    tv_list = []
    for i in range(0, 300, 20):
        url = 'https://movie.douban.com/j/search_subjects?type=tv&tag=%E7%83%AD%E9%97%A8&sort=recommend&page_limit=20&page_start={}'.format(i)
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"}
        response = requests.get(url, headers = headers)
        html_data = response.text
        data = json.loads(html_data)
        data_list = data['subjects']
        for tv in data_list:
            title = tv['title']
            url = tv['url']
            rate = tv['rate']
            tv_list.append({
                'title': title,
                'url': url,
                'rate': rate
                })
    pd.DataFrame(tv_list).to_csv("douban.csv", index = False)
    
    
if __name__ == '__main__':
    spider()
