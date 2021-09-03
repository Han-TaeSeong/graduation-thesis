# BeautifulSoup를 이용한 크롤링
from bs4 import BeautifulSoup
import requests
import csv

# response = requests.get("https://finance.naver.com/")
# html = response.text
# soup = BeautifulSoup(html,'html.parser')
#
#
# title = soup.select_one('#content > div.article > div.section > div.news_area > div > ul > li:nth-child(1) > span')
#
# print(title)

########## 자동로그인
import time
# from selenium import webdriver
# driver = webdriver.Chrome()
# driver.get('https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com')
# time.sleep(2)
# driver.find_element_by_xpath('//*[@id="id"]').send_keys('tjd6487')
# driver.find_element_by_name('pw').send_keys('7552123a^^^')
# driver.find_element_by_id('log.login').click()


from selenium import webdriver
from bs4 import BeautifulSoup
# driver = webdriver.Chrome()
# driver.get('https://www.naver.com/')
# html = driver.page_source
# soup = BeautifulSoup(html,'html.parser')
# notices = soup.select('#NM_FAVORITE > div.group_nav')
#
# for i in notices:
#     print(i.text.strip())

# from urllib.request import urlopen
# html = urlopen('https://finance.naver.com/item/main.nhn?code=005930')
# bsObject = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
# expected_per = bsObject.find('em', id='_cns_per')
# expected_per = expected_per.get_text()
# print(expected_per)
# expected_pbr = bsObject.find('em',id='_pbr')
# expected_pbr = expected_pbr.get_text()
# print(expected_pbr)

# from pykrx import stock
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# from urllib.request import urlopen
# from bs4 import BeautifulSoup
# from selenium import webdriver
# import pandas as pd
# import matplotlib.pyplot as plt
#
# #3년전 및 오늘
# start = datetime.today() + relativedelta(years=-3)
# end = datetime.today()
#
# #3년전 및 오늘 문자열로 양식 지정
# start = start.strftime("%Y")
# end = end.strftime("%Y")
#
# # 2018년-2020년 DIV/BPS/PER/EPS 조회(임의의 코드로도 불러올 수 있도록 코드 바꿔야함)
# df = stock.get_market_fundamental_by_date(start, end, "005930", freq='y')
#
# # PER,PBR,DIV,DPS만 추출
# df = df[['PER', 'PBR', 'DIV']]
# df.rename(columns={'DIV':'DIV(%)'}, inplace=True)
# df.index = ['2018', '2019', '2020']
#
# # 올해 추정 PER, PBR 가져오기(html 주소에서 종목코드 임의로 받을 수 있도록 변경해야함 현재는 삼성전자코드 005930을 활용), (pbr의 경우 추정데이터를 찾을 수 없으므로 올해 6월 데이터를 기준으로함)
# html = urlopen('https://finance.naver.com/item/main.nhn?code=005930')
# bsObject = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
# expected_per = bsObject.find('em', id='_cns_per')
# expected_per = expected_per.get_text()
# expected_pbr = bsObject.find('em', id='_pbr')
# expected_pbr = expected_pbr.get_text()
#
# # 기존 df에 expected_per, 올해 6월 기준 pbr 합치기(div,dps는 미기재)
# expected_data = {
#     'PER' : [expected_per],
#     'PBR' : [expected_pbr],
#     'DIV(%)' : ['']
# }
# expected_df = pd.DataFrame(expected_data)
#
# df = pd.concat([df, expected_df])
#
#
# # KOSDAQ 및 KOSPI의 PER, PBR, DIV 불러오기
# driver = webdriver.Chrome()
# driver.implicitly_wait(5)
# driver.get('https://kosis.kr/statHtml/statHtml.do?orgId=343&tblId=DT_343_2010_S0033')
#
# KOSPI_PER = driver.find_element_by_xpath('/html/body/form/div[2]/div[5]/div[4]/div[5]/div[1]/table/tbody/tr[1]/td[4]/span[2]')
# KOSPI_PER = KOSPI_PER.text
#
# driver.get('https://kosis.kr/statHtml/statHtml.do?orgId=343&tblId=DT_343_2010_S0073')
#
# KOSDAQ_PER = driver.find_element_by_xpath('/html/body/form/div[2]/div[5]/div[4]/div[5]/div[1]/table/tbody/tr[1]/td[4]/span[2]')
# KOSDAQ_PER = KOSDAQ_PER.text
#
# driver.get('https://kosis.kr/statHtml/statHtml.do?orgId=343&tblId=DT_343_2010_S0034')
#
# KOSPI_PBR = driver.find_element_by_xpath('/html/body/form/div[2]/div[5]/div[4]/div[5]/div[1]/table/tbody/tr[1]/td[4]')
# KOSPI_PBR = KOSPI_PBR.text
#
# driver.get('https://kosis.kr/statHtml/statHtml.do?orgId=343&tblId=DT_343_2010_S0074')
#
# KOSDAQ_PBR = driver.find_element_by_xpath('//*[@id="mainTable"]/tbody/tr[1]/td[4]/span[2]')
# KOSDAQ_PBR = KOSDAQ_PBR.text
#
# driver.get('https://kosis.kr/statHtml/statHtml.do?orgId=343&tblId=DT_343_2010_S0032')
#
# KOSPI_DIV = driver.find_element_by_xpath('/html/body/form/div[2]/div[5]/div[4]/div[5]/div[1]/table/tbody/tr[1]/td[4]')
# KOSPI_DIV = KOSPI_DIV.text
#
# driver.get('https://kosis.kr/statHtml/statHtml.do?orgId=343&tblId=DT_343_2010_S0072')
#
# KOSDAQ_DIV = driver.find_element_by_xpath('/html/body/form/div[2]/div[5]/div[4]/div[5]/div[1]/table/tbody/tr[1]/td[4]/span[2]')
# KOSDAQ_DIV = KOSDAQ_DIV.text
#
# # 수정된 df에 KOSPI, KOSDAQ 데이터 추가하기
# KOSPI_data = {
#     'PER' : [KOSPI_PER],
#     'PBR' : [KOSPI_PBR],
#     'DIV(%)' : [KOSPI_DIV]
# }
#
# KOSDAQ_data = {
#     'PER' : [KOSDAQ_PER],
#     'PBR' : [KOSDAQ_PBR],
#     'DIV(%)' : [KOSDAQ_DIV]
# }
#
# KOSPI_df = pd.DataFrame(KOSPI_data)
# KOSDAQ_df = pd.DataFrame(KOSDAQ_data)
# df = pd.concat([df, KOSPI_df])
# df = pd.concat([df, KOSDAQ_df])
# df.index = ['2018', '2019', '2020', '2021', 'KOSPI 2021E', 'KOSDAQ 2021E']
# df = df.transpose()
#
# #########################################
# df.iloc[2,3]=1
# df=df.astype(float)
# ################################
#
#
# print(df)
# print(df.columns)
#
# # 데이터 시각화 (데이터프레임 칼럼명이 x축, y축은 수치값)
# ax = df.plot(kind = 'bar', rot=0, figsize=(16, 6))
# plt.show()