"""
This file is adapted from https://github.com/glhuilli/neurips_crawler/blob/master/src/neurips_crawler.py
"""

import argparse
import json
import logging
import os
import re
import time
import uuid
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, TextIO

from socket import error as socket_error
import errno

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm



_BASE_URL = 'http://papers.nips.cc'
_CRAWLING_WAIT_TIME = 0.3
_NEURIPS_NAMESPACE = uuid.UUID('5ee6531f-0d79-4cf1-8da6-dc83cb553336')
_FIRST_YEAR = 1988
_PDF_FOLDER = 'pdfs'
_OUTPUT_PAPERS_FILE = 'papers_data.jsons'


class NeuripsUrl(NamedTuple):
    """
    Immutable object that stores a NeurIPS conference url and year
    """
    url: str
    year: str


class NeuripsPaperData(NamedTuple):
    """
    Immutable object that stores a NeurIPS paper data
    """
    id_: str
    title: str
    pdf_name: str
    pdf_link: str
    info_link: str


class NeuripsPaper:
    """
    Mutable object that stores all relevant properties from a paper
    """

    def __init__(self, data: NeuripsPaperData):
        self.data = data
        self.abstract: Optional[str] = None
        self.authors: Optional[List[Dict[str, str]]] = None

    def to_json(self):
        """
        Excludes pdf_link and info_link as it's not needed for further data analysis.
        """
        return {
            'id': self.data.id_,
            'title': self.data.title,
            'pdf_name': self.data.pdf_name,
            'abstract': self.abstract,
            'authors': self.authors
        }


def crawl_papers(neurips_url: NeuripsUrl, logger: logging.Logger, downloaded_files: Set[str]) -> Iterable[NeuripsPaper]:
    """
    Iterate over each paper found in conference url.
    """
    for link in get_paper_links(neurips_url.url):
        time.sleep(_CRAWLING_WAIT_TIME)
        pdf_name = link['href'].split('/')[-1] + '.pdf'
        if pdf_name in downloaded_files:
            logger.info(f'skipping {pdf_name}')
            continue
        paper = init_neurips_paper(link)
        try:
            paper_soup = BeautifulSoup(requests.get(paper.data.info_link).content, 'lxml')
            get_abstract(paper, paper_soup)
            get_authors(paper, paper_soup)
            yield paper
        except requests.exceptions.ConnectionError:
            logger.info('Exception when crawling paper: %s', paper.to_json(), exc_info=True)
        except socket_error as e:
            if e.errno != errno.ECONNRESET:
                raise
            pass


def get_conference_links(year_from: int, year_to: int) -> Iterable[NeuripsUrl]:
    """
    Return the link to crawl for each NeurIPS conference between input years.
    """
    base = _BASE_URL + '/book/advances-in-neural-information-processing-systems-{}-{}'
    number_year_from = year_from - _FIRST_YEAR + 1
    number_year_to = year_to - _FIRST_YEAR + 1
    for idx, i in enumerate(range(number_year_from, number_year_to + 1)):
        year = str(year_from + idx)
        url = base.format(str(i), year)
        yield NeuripsUrl(url=url, year=year)


def save_pdf_file(pdf_link: str, pdf_name: str, year_output_folder: str) -> None:
    """
    Downloads the paper and saves it to the pdf folder.
    """
    pdf = requests.get(pdf_link)
    with open(os.path.join(year_output_folder, _PDF_FOLDER, pdf_name), 'wb') as pdf_file:
        pdf_file.write(pdf.content)


def get_paper_links(url: str) -> Iterable[BeautifulSoup]:
    """
    Locates all paper links in the url and returns a BeautifulSoup object
    """
    try:
        url_request = requests.get(url)
    except requests.exceptions.ConnectionError:
        url_request = requests.get(url, proxies=dict(http='socks5://127.0.0.1:1080',
                                                     https='socks5://127.0.0.1:1080'))
    for link in BeautifulSoup(url_request.content, 'lxml').find_all('a'):
        if link['href'][:7] == '/paper/':
            yield link


def init_neurips_paper(link: BeautifulSoup) -> NeuripsPaper:
    """
    Given the BeautifulSoup object, initializes a NeuripsPaper
    """
    paper_title = link.contents[0]
    info_link = _BASE_URL + link['href']
    pdf_link = info_link + '.pdf'
    pdf_name = link['href'][7:] + '.pdf'
    paper_id = str(uuid.uuid5(_NEURIPS_NAMESPACE, re.findall(r'^(\d+)-', pdf_name)[0]))
    return NeuripsPaper(
        data=NeuripsPaperData(
            id_=paper_id,
            title=paper_title,
            info_link=info_link,
            pdf_link=pdf_link,
            pdf_name=pdf_name))


def get_abstract(paper: NeuripsPaper, paper_soup: BeautifulSoup) -> None:
    """
    Searches for a paragraph of class 'abstract' and adds it to the paper object

    Note that the abstract might not be always available in the web page
    (sometimes it's just a "Abstract Missing" message)
    """
    paper.abstract = paper_soup.find('p', attrs={'class': 'abstract'}).contents[0]


def get_authors(paper: NeuripsPaper, paper_soup: BeautifulSoup) -> None:
    """
    Finds all authors in the BeautifulSoup object.

    The author_id is a permalink used for a given author across all conferences.
    """
    paper.authors = []
    for author in paper_soup.find_all('li', attrs={'class': 'author'}):
        author_id = author.contents[0]['href'].split('/author/')[-1]
        author_name = author.contents[0].contents[0]
        paper.authors.append({'id': author_id, 'name': author_name})


def save_paper(paper: NeuripsPaper, output_folder: str, output: TextIO,
               logger: logging.Logger) -> None:
    """
    Saves PDF and writes the NeuripsPaper json into the output TextIO
    """
    save_pdf_file(paper.data.pdf_link, paper.data.pdf_name, output_folder)
    try:
        output.write(json.dumps(paper.to_json()) + '\n')
    except TypeError:
        logger.info('Exception saving paper: %s', paper.__dict__, exc_info=True)


def get_logger(log_file: str) -> logging.Logger:
    """
    This logger prints both in screen and into a log file at the same time
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    """
    argparse initializer. Note that only output and log have default values.
    """
    parser = argparse.ArgumentParser(description='Crawl NeurIPS Papers.')
    parser.add_argument('--from_year', help='Starting year to crawl')
    parser.add_argument('--to_year', help='Final year to crawl')
    parser.add_argument('--output', default='./output/', help='Output path')
    parser.add_argument('--log', default='./crawler_log.txt', help='Log file')
    parser.add_argument('--force', action='store_true', default=False, help='force run already downloaded years')
    return parser.parse_args()


def main(args):
    """
    Iterates over conferences from input years and then papers within these conferences.
    Saves both pdf from papers found and a Json representation of a paper's information.
    If year folder is available, then it will skip that year from crawling.

    Exceptions will be logged into a log file and displayed in the console.
    """
    logger = get_logger(args.log)
    logger.info('crawling NeurIPS')
    for neurips_url in tqdm(
            get_conference_links(int(args.from_year), int(args.to_year)),
            'iterating over conferences'):
        logger.info(f'NeurIPS-{neurips_url.year} crawling started.')
        year_output_folder = os.path.join(args.output, f'data_{neurips_url.year}')
        files = set()
        papers_meta_data = {}
        if args.force:
            #  load data from year metadata if available to skip files already downloaded
            if os.path.isdir(year_output_folder):
                with open(os.path.join(year_output_folder, _OUTPUT_PAPERS_FILE), 'r') as mdf:
                    for line in mdf.readlines():
                        meta_data = json.loads(line)
                        papers_meta_data[meta_data['pdf_name']] = meta_data
                        files.add(meta_data['pdf_name'])
        logger.info(f'already processed: {len(files)}')

        if os.path.isdir(year_output_folder) and not args.force:
            logger.info(f'Year {neurips_url.year} was already processed.')
            continue

        if not os.path.isdir(year_output_folder):  # create folder if year_output_folder doesn't exist
            os.makedirs(f'{year_output_folder}/{_PDF_FOLDER}')

        with open(os.path.join(year_output_folder, _OUTPUT_PAPERS_FILE), 'a') as output:
            for paper in tqdm(crawl_papers(neurips_url, logger, downloaded_files=files), 'iterating over papers'):
                save_paper(paper, year_output_folder, output, logger)
        logger.info(f'NeurIPS-{neurips_url.year} crawled successfully.')
        time.sleep(_CRAWLING_WAIT_TIME)


if __name__ == '__main__':
    os.chdir("../../..")
    main(parse_args())