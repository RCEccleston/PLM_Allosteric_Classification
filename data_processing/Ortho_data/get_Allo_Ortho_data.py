from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

driver.get('https://mdl.shsmu.edu.cn/ASD2023Common/allosite')

all_pdb_links = set()

while True:
    time.sleep(1)
    rows = driver.find_elements(By.CSS_SELECTOR, 'table tbody tr')
    for row in rows:
        link_elements = row.find_elements(By.CSS_SELECTOR, 'a')
        for a in link_elements:
            href = a.get_attribute('href')
            if href and href.endswith('.pdb.gz'):
                all_pdb_links.add(href)

    try:
        next_button = driver.find_element(By.LINK_TEXT, '›')
        if 'disabled' in next_button.get_attribute('class'):
            break
        next_button.click()
    except:
        break

driver.quit()

# Save links to a file
with open('pdb_links.txt', 'w') as f:
    for link in sorted(all_pdb_links):
        f.write(link + '\n')
