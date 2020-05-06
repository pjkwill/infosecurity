#2017302242pjk

import re
import requests
import xlwt
#excel库 xlwt

#观察当当网相关url得到通用规律
url_basic = 'http://search.dangdang.com/?key='
heads = {
  #用户代理
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36\
    (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
}

yucun = []
for i in range(5):  #括号内写入需要爬取的具体页数
    try:
        key = "信息安全"  #输入想查询的关键词
        url = url_basic + key + "&act=input&page_index=" + str(i + 1)#设定规范的url
        print(url)
        response = requests.get(url, headers=heads)
        content = response.text
        pattern = re.compile(
            '<li.*?<a title="(.*?)".*?>.*?search_now_price">.*?(\d+\D\d+)</span>.*?search_pre_price">.*?(\d+\D\d+)</span>.*?<a href=.*?ddclick=.*?>(\d+).*?</a>.*?<a href=.*?>(.*?)</a>.*?</span>.*?</li>',
            re.S)
        results = re.findall(pattern, content)
        yucun += results
        print("获取成功")#如果可以正常输出，则可以看到情况
        i += 1#继续向目标页数爬取
    except:
        break#实现所需页码后跳出循环

biaotou = ["序号", "书名","现价", "原价", "评论数" ,"作者" ]
with open(r"D:\python.xlsx", "w") as file:
    file = xlwt.Workbook()
    data = file.add_sheet('sheet_1')
    data.write(0, 0, biaotou[0])
    data.write(0, 1, biaotou[1])
    data.write(0, 2, biaotou[2])
    data.write(0, 3, biaotou[3])
    data.write(0, 4, biaotou[4])
    data.write(0, 5, biaotou[5])
    i = 0
    s = 1
    for result in yucun:
        data.write(i + 1, 0, s)
        data.write(i + 1, 1, result[0])
        data.write(i + 1, 2, result[1])
        data.write(i + 1, 3, result[2])
        data.write(i + 1, 4, result[3])
        data.write(i + 1, 5, result[4])
        s += 1
        i += 1
file.save(r"D:\python.xlsx")
print("数据已保存")

