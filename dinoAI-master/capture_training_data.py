from selenium import webdriver
import selenium.webdriver.common.keys as Keys
import time
import threading
from capture import capture_feed
from selenium.webdriver.chrome.options import Options

options = Options()
#options.add_argument('--headless')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(executable_path=r'/usr/bin/chromedriver',chrome_options=options)
# internet connection must be off
driver.get('chrome://dino')
time.sleep(2)
page = driver.find_element_by_class_name('offline')
dino = driver.find_element_by_class_name("runner-container")
location = dino.location
size = dino.size
print(location)
print(size)
page.send_keys(u'\ue00d')

capture_feed.start(driver,location)

driver.close()



