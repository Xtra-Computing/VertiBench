import os
import time
import argparse

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains

ICLR_categories = {
    2019: ['poster-presentations', 'oral-presentations', 'withdrawn-rejected-submissions'],
    2020: ['poster-presentations', 'spotlight-presentations', 'oral-presentations', 'withdrawn-rejected-submissions'],
    2021: ['poster-presentations', 'spotlight-presentations', 'oral-presentations', 'withdrawn-rejected-submissions'],
    2022: ['poster-submissions', 'spotlight-submissions', 'oral-submissions', 'submitted-submissions', 'desk-rejected-withdrawn-submissions'],
    2023: ['notable-top-5-', 'notable-top-25-', 'poster', 'submitted', 'desk-rejected-withdrawn-submissions']
}

ICLR_category_to_label = {
    'poster-presentations': 'accept',
    'spotlight-presentations': 'accept',
    'oral-presentations': 'accept',
    'poster-submissions': 'accept',
    'spotlight-submissions': 'accept',
    'oral-submissions': 'accept',
    'notable-top-5-': 'accept',
    'notable-top-25-': 'accept',
    'poster': 'accept',

    'withdrawn-rejected-submissions': 'reject',
    'submitted-submissions': 'reject',
    'desk-rejected-withdrawn-submissions': 'reject',
    'submitted': 'reject',
}

NeurIPS_categories = {
    2021: ['oral-presentations', 'spotlight-presentations', 'poster-presentations', 'rejected-papers-opted-in-public'],
    2022: ['accepted-papers', 'rejected-papers-opted-in-public'],
}


def get_paper_list(conference, years, save_dir):
    driver = webdriver.Chrome()
    if isinstance(years, int):
        years = [years]

    os.makedirs(save_dir, exist_ok=True)

    for year in years:
        save_path = os.path.join(save_dir, f'{conference}{year}-list.tsv')  # tab-separated file to adapt comma in title
        with open(save_path, 'w') as f:
            f.write('paper_id\ttitle\tauthors\tpdf_url\tforum_url\tn_replies\tcategory\n')

            if conference == 'iclr':
                categories = ICLR_categories[year]
                conf_id = 'ICLR.cc'
            elif conference == 'neurips':
                categories = NeurIPS_categories[year]
                conf_id = 'NeurIPS.cc'

            for category in categories:
                driver.get(f"https://openreview.net/group?id={conf_id}/{year}/Conference#{category}")
                driver.maximize_window()
                driver.refresh()

                # check if the page is loaded
                loaded_condition = EC.presence_of_element_located((By.XPATH, f"//div[@class='tab-content']/div/ul/li[@class='note ']"))
                WebDriverWait(driver, 50).until(loaded_condition)

                try:
                    end_page_btn = driver.find_element(By.XPATH, f"//div[@id='{category}']/nav/ul[@class='pagination']/li[@class='  right-arrow']/a[text()='»']/..")
                    num_pages = int(end_page_btn.get_attribute('data-page-number'))
                except NoSuchElementException:
                    num_pages = 1

                print(f"Category {category}. Number of pages: {num_pages}")

                for page in range(num_pages):
                    # check if the page is loaded
                    loaded_condition = EC.presence_of_element_located((By.XPATH, f"//div[@class='tab-content']/div/ul/li[@class='note ']"))
                    WebDriverWait(driver, 50).until(loaded_condition)

                    submissions_on_page = driver.find_element(By.ID, category).find_elements(By.CLASS_NAME, 'note')
                    for submission in tqdm(submissions_on_page, desc=f"Category {category}, Page {page}"):
                        try:
                            title_h4 = submission.find_element(By.TAG_NAME, 'h4')
                            title = title_h4.find_elements(By.TAG_NAME, 'a')[0].text.strip()
                            forum_url = title_h4.find_elements(By.TAG_NAME, 'a')[0].get_attribute('href')
                            pdf_url = title_h4.find_elements(By.TAG_NAME, 'a')[1].get_attribute('href')
                            paper_id = pdf_url.split('=')[-1]

                            author_div = submission.find_element(By.CLASS_NAME, 'note-authors')
                            authors = [author.text.strip() for author in author_div.find_elements(By.TAG_NAME, 'a')]
                            authors_str = ','.join(authors)

                            meta_div = submission.find_element(By.CLASS_NAME, 'note-meta-info')
                            n_replies = meta_div.find_elements(By.TAG_NAME, 'span')[-1].text.split(' ')[0]

                            line = f"{paper_id}\t{title}\t{authors_str}\t{pdf_url}\t{forum_url}\t{n_replies}\t{category}\n"
                            f.write(line)
                        except Exception as e:
                            print(e)
                            continue

                    try:
                        next_page_btn = driver.find_element(By.XPATH, f"//div[@id='{category}']/nav/ul[@class='pagination']/li[@class='  right-arrow']/a[text()='›']")
                        if next_page_btn is None:
                            print(f"Page {page}. No more pages.")
                            break
                        else:
                            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            next_page_btn.click()
                            time.sleep(5)
                    except NoSuchElementException:
                        print(f"Page {page}. No more pages.")
                        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', type=int, nargs='+')
    parser.add_argument('-s', '--save_dir', type=str, default='_data/real/paper/raw/')
    parser.add_argument('-c', '--conference', type=str, default='iclr', help='[iclr, neurips]')
    args = parser.parse_args()

    assert args.conference in ['iclr', 'neurips']

    get_paper_list(args.conference, args.year, args.save_dir)

