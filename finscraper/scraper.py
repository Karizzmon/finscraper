import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


def get_10k_urls(cik, num_docs=10):
    base_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=exclude&count={num_docs}"
    response = requests.get(
        base_url, headers={"User-Agent": "Your Name yourname@email.com"}
    )
    soup = BeautifulSoup(response.content, "html.parser")
    doc_links = soup.find_all("a", {"id": "documentsbutton"})
    return [f"https://www.sec.gov{link['href']}" for link in doc_links]


def download_10k(url, save_dir):
    response = requests.get(
        url, headers={"User-Agent": "Chirag Aggarwal chiragaggarwal5k@email.com"}
    )
    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.find("table", {"class": "tableFile"})
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) > 3 and cells[3].text == "10-K":
            doc_url = f"https://www.sec.gov{cells[2].a['href']}"
            doc_response = requests.get(
                doc_url,
                headers={"User-Agent": "Chirag Aggarwal chiragaggarwal5k@email.com"},
            )
            filename = os.path.join(save_dir, f"{url.split('/')[-1]}.html")
            with open(filename, "wb") as f:
                f.write(doc_response.content)
            print(f"Downloaded: {filename}")
            break


def main():
    cik = "0000320193"  # Apple Inc.
    save_dir = "financial_statements"
    os.makedirs(save_dir, exist_ok=True)

    urls = get_10k_urls(cik)
    for url in urls:
        download_10k(url, save_dir)


if __name__ == "__main__":
    main()
