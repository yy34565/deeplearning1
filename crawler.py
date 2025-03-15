from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--headless')  # 可以设置无界面模式，可不加这行
driver = webdriver.Chrome(options=chrome_options)
url = "https://example.com/dynamic_page"
driver.get(url)
# 等待页面动态加载完成，可以用显式等待或隐式等待
# 例如隐式等待
driver.implicitly_wait(10)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
# 再进行数据提取等操作
driver.quit()
