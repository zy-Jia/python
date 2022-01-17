import json
import requests
from lxml import html
import pandas as pd

headers = {"User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15"}
url = "https://api.inews.qq.com/newsqa/v1/query/inner/publish/modules/list?modules=provinceCompare"
response = requests.get(url, headers = headers)
print(response.status_code)
html_data = response.text
#print(html_data)
html_data = json.loads(html_data)
#print(html_data)
data = html_data["data"]
#print(data)
provinceCompare = data["provinceCompare"]
#print(provinceCompare)
province_list = []
for province_name in provinceCompare:
    #print(province_name)
    province_msg = provinceCompare[province_name]
    #print(province_msg)
    nowConfirm = province_msg["nowConfirm"]
    confirmAdd = province_msg["confirmAdd"]
    dead = province_msg["dead"]
    heal = province_msg["heal"]
    zero = province_msg["zero"]
    province_list.append({
        "provinceName":province_name,
        "nowConfirm":nowConfirm,
        "confirmAdd":confirmAdd,
        "dead":dead,
        "heal":heal,
        "zero":zero
    })
pd.DataFrame(province_list).to_csv("province.csv",index=False)
url = "https://api.inews.qq.com/newsqa/v1/query/inner/publish/modules/list?modules=chinaDayList"
response = requests.get(url, headers = headers)
#print(response.status_code)
html_data = response.text
#print(html_data)
html_data = json.loads(html_data)
#print(html_data)
data = html_data["data"]
#print(data)
chinaDayList = data["chinaDayList"]
print(chinaDayList)
daily_msg_list = []
for daily_msg in chinaDayList:
    importedCase = daily_msg["importedCase"]
    suspect = daily_msg["suspect"]
    dead = daily_msg["dead"]
    heal = daily_msg["heal"]
    confirm = daily_msg["confirm"]
    nowConfirm = daily_msg["nowConfirm"]
    date = daily_msg["date"]
    date = "2021-" + date.replace(".","-")
    print(date)
    daily_msg_list.append({
        "importedCase":importedCase,
        "suspect":suspect,
        "dead":dead,
        "heal":heal,
        "confirm":confirm,
        "nowConfirm":nowConfirm,
        "date":date
    })
pd.DataFrame(daily_msg_list).to_csv("daily.csv",index=False)