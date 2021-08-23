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

import time
from selenium import webdriver
driver = webdriver.Chrome()
driver.get('https://finance.naver.com/')
time.sleep(2)
driver.find_element_by_name()


