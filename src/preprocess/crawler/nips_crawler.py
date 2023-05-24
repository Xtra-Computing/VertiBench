import argparse
import json
import logging
import os
import re
import time
import uuid
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, TextIO

import socks
from socket import error as socket_error
import socket
import errno

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_paper_list(years, save_dir):
    if isinstance(years, int):
        years = [years]

    os.makedirs(save_dir, exist_ok=True)

    for year in years:
        document = requests.get(f"https://proceedings.neurips.cc/paper_files/paper/{year}")
        soup = BeautifulSoup(document.text, 'lxml')
        paper_list = soup.find('ul', {'class': 'paper-list'}).find_all('li')

        save_path = os.path.join(save_dir, f'neurips{year}-accept-list.tsv')

        with open(save_path, 'w') as f:
            f.write('paper_id\ttitle\tauthors\tforum_url\tcategory\n')
            for paper_li in paper_list:
                title = paper_li.find('a').text.strip()
                forum_url = paper_li.find('a').get('href')
                paper_id = forum_url.split('/')[-1].split('-')[0]
                authors = paper_li.find('i').text.split(', ')
                authors_str = ','.join(authors)

                # if the class of paper_li is 'none' or 'conference', it means the paper is in main conference track
                # if the class of paper_li is 'datasets_and_benchmarks', it means the paper is in dataset track
                if paper_li.get('class') == ['datasets_and_benchmarks']:
                    category = 'dataset'
                elif paper_li.get('class') in [['none'], ['conference']]:
                    category = 'main'
                else:
                    raise ValueError(f"Unknown paper category: {paper_li.get('class')}")
                f.write(f'{paper_id}\t{title}\t{authors_str}\t{forum_url}\t{category}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--years', type=int, nargs='+')
    parser.add_argument('-s', '--save_dir', type=str, default='_data/real/paper/raw/')
    args = parser.parse_args()

    get_paper_list(args.years, args.save_dir)


