import os
import urllib.request
import pandas as pd
import requests
from bs4 import BeautifulSoup


def download_all_pdfs(path, link):
    """
    Gathers all links on the input website and downloads the pdfs.
    :param path: path to save the files to
    :param link: initial link to download the pdfs from
    """
    path_link = path + link.split('//')[-1].split('/')[0].split('.')[0]
    if not os.path.exists(path_link):
        os.mkdir(path_link)
    links = pd.Series(retrieve_links(link))
    links_pdf = pd.concat([links[links.str.contains('somo.nl/download/')], links[links.str.endswith('.pdf')]]) # only keep pdfs
    links_pdf.apply(lambda link: download_file(link, path_link))

    # retrieve sublinks of the links that are not a pdf
    for i in range(2):
        links = links[~links.str.contains('somo.nl/download/')]
        links = links[~links.str.endswith('.pdf')]
        links = links.apply(retrieve_links)
        links = links.explode().drop_duplicates().dropna()

        links_pdf = pd.concat(
            [links[links.str.contains('somo.nl/download/')], links[links.str.endswith('.pdf')]]).drop_duplicates()
        links_pdf.apply(lambda link: download_file(link, path_link))


def retrieve_links(link):
    """
    Extracts all links from the input website.
    :param link: input website
    :return: list of links on the website
    """
    try:
        response = requests.get(link)
        html_page = BeautifulSoup(response.text, 'html.parser')
        links = pd.Series(html_page.find_all('a'))
        links = links.apply(lambda link: link.get('href'))
        links = links[links.str.contains('http')].to_list()
        return links
    except:
        return None


def download_file(link, path):
    """
    Downloads the file given by the link path and saves it to the path.
    :param link: link to file to download
    :param path: path to save the file to
    """
    name = link.split('/')[-1]
    try:
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', 'Mozilla/5.0')
        opener.retrieve(link, path + '/' + name)
    except:
        try:
            response = requests.get(link)
            pdf = open(name, 'wb')
            pdf.write(response.content)
            pdf.close()
        except:
            try:
                print(link)
            except:
                print('cannot print')


if __name__ == '__main__':
    # List of the links to the relevant NGO website to download the articles from in addition to scientific articles
    my_urls = ['https://asia.floorwage.org/reports/', 'https://globallaborjustice.org/publications/',
               'https://labourbehindthelabel.org/resources/reports/'
               'https://cleanclothes.org/resources/reports-archive',
               'https://somo.nl/latest-updates/?post_date_min=2017-01-01&post_date_max=&post-format%5B%5D=publication', ]

    path_to_pdfs = ''

    for my_url in my_urls:
        download_all_pdfs(path_to_pdfs, my_url)
