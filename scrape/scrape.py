from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import sys
import time

cat = str(sys.argv[1])

driver = webdriver.Chrome('/home/yaniv/Downloads/chromedriver')
driver.get('https://www.google.co.il/imghp?hl=en&ogbl')

box = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
box.send_keys(cat + ' in the wild')
box.send_keys(Keys.ENTER)


last_height = driver.execute_script('return document.body.scrollHeight')
while True:
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(2)
    new_height = driver.execute_script('return document.body.scrollHeight')
    try:
        driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
        time.sleep(2)
    except:
        pass
    if new_height == last_height:
        break
    last_height = new_height

time.sleep(10)

i = 1
for i in range(1, 600):
    try: 
        pic = driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div[%d]/a[1]/div[1]/img' % (i))
        pic.screenshot('/home/yaniv/Documents/big cats/%s/%s-%d.png' % (cat, cat, i))
    except:
        pass

print(i)

