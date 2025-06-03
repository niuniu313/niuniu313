from DrissionPage import ChromiumPage
from DrissionPage.common import Actions
import csv
import time

f = open('jd_data.csv',mode = 'w',encoding = 'utf-8-sig',newline='')
csv_writer = csv.DictWriter(f , fieldnames=[
    '昵称',
    '日期',
    '产品',
    '分数',
    '评论'
])
csv_writer.writeheader()

dp = ChromiumPage()
ac = Actions(dp)
dp.get('https://item.jd.com/100124527967.html')
dp.listen.start('client.action')
dp.ele('css:.arrow').click()

for i in range(50):
    tab = dp.ele('css:div._rateListContainer_1ygkr_45')
    tab.scroll.down(i * 800)

    r = dp.listen.wait()
    json_data = r.response.body
    comments = json_data['result']['floors'][2]['data']

    time.sleep(1)


    for comment in comments:
        try:
            dit= {
                '昵称':comment['commentInfo']['userNickName'],
                '日期': comment['commentInfo']['commentDate'],
                '产品': comment['commentInfo']['productSpecifications'][3:],
                '分数': comment['commentInfo']['commentScore'],
                '评论': comment['commentInfo']['commentData'],
            }
            csv_writer.writerow(dit)
            print(dit)




        except Exception as e:
            pass