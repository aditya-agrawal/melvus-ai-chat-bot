from bs4 import BeautifulSoup

# import nltk
# nltk.download('words')
# words = set(nltk.corpus.words.words())

import pdfkit
import requests
import re
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urljoin


class BeautifulSoupFlexibleScraper:
    """
    A web scraper that uses BeautifulSoup to scrape web pages and retrieve child URLs, sort them by hierarchy,
    and extract text content from the pages.

    Attributes:
    url_parent (str): The parent URL from which child URLs are extracted.
    url_children (list[str]): A list of child URLs found within the parent URL.
    len_url_children (int): The number of child URLs found within the parent URL.
    count_processed_urls (int): The number of processed URLs.
    count_none_pages (int): The number of pages that could not be scraped.
    len_current_content (int): The length of the content for the current web page.
    len_total_content (int): The total length of the content scraped so far.

    """

    def __init__(self):
        """
        Constructs a new BeautifulSoupFlexibleScraper object.

        Parameters:
        -----------
        None
        """
        self.url_parent = None
        self.url_children = None
        self.len_url_children = None
        self.count_processed_urls = 0
        self.count_none_pages = 0
        self.len_current_content = None
        self.len_total_content = 0

    def get_child_urls(self, url_parent, depth=1, max_workers=10):
        if depth < 1:
            raise ValueError("Depth must be a positive integer.")

        self.url_parent = url_parent
        child_urls = [url_parent]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for d in range(depth):
                future_to_url = {
                    executor.submit(self.get_direct_child_urls, url): url
                    for url in child_urls
                }
                child_urls = []
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print("%r generated an exception: %s" % (url, exc))
                    else:
                        child_urls.extend(data)

        child_urls = [*set(child_urls)]
        child_urls = self.sort_urls_by_hierarchy(child_urls)

        self.url_children = child_urls
        self.len_url_children = len(child_urls)
        return child_urls

    def get_direct_child_urls(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a")
        child_urls = []
        for link in links:
            href = link.get("href")
            if href:
                abs_url = urljoin(url, href)
                if abs_url.startswith(self.url_parent):
                    child_urls.append(abs_url)
        return child_urls

    def sort_urls_by_hierarchy(self, urls):
        """
        Sort a list of URLs by their hierarchy, i.e. their domain name and path.

        Args:
            urls (list): A list of URLs as strings.

        Returns:
            A sorted list of URLs as strings.
        """
        parsed_urls = [urlparse(url) for url in urls]

        # Sort by domain name
        sorted_urls = sorted(parsed_urls, key=lambda x: x.netloc)

        # Sort by path, but within each domain, sort by the length of the path
        sorted_urls = sorted(sorted_urls, key=lambda x: (x.netloc, len(x.path)))

        # Convert back to strings
        sorted_urls = [url.geturl() for url in sorted_urls]

        return sorted_urls

    def scrape_page(self, url):
        """
        Scrapes a web page and returns its text content.

        Parameters:
        -----------
        url : str
            The URL of the web page to scrape.

        Returns:
        --------
        str:
            The text content of the web page, with non-English words, special characters, and empty new lines removed.

        """
        try:
            # Make a request to the specified URL
            response = requests.get(url)

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract the text content of the page
            text = soup.get_text()

            # Find images in the page, extract and save them
            images = soup.find_all("img")
            image_sources = []
            for image in images:
                image_sources.append(image.get("src"))

            # Remove non-English words, special characters, and empty new lines
            text = re.sub(r"[^\w\s]|\d", "", text)
            text = re.sub(r"\b(?![a-zA-Z])[^\W_]+\b", "", text)
            text = re.sub(r"\n\s*\n", "\n", text)
            # text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())

            self.count_processed_urls += 1
            self.len_current_content = len(text)
            self.len_total_content += len(text)

            return text, image_sources

        except:
            # Return None if the page could not be scraped
            self.count_processed_urls += 1
            self.count_none_pages += 1
            self.len_current_content = None
            return None

    def save_page_as_pdf(self, url, content, output_path):
        """
        Saves the given content as a PDF file to the specified output path.
        """
        # Assuming content is HTML or suitable text for PDF conversion
        pdfkit.from_string(content, output_path)


if __name__ == "__main__":
    scraper = BeautifulSoupFlexibleScraper()
    url_parent = "https://milvus.io/docs"
    start_time = time.time()
    all_urls = scraper.get_child_urls(url_parent=url_parent, depth=2)
    end_time = time.time()
    print(all_urls)
    print(f"Length of URLs to scrape: {len(all_urls)}")
    print(f"Process took {end_time - start_time} seconds to complete")
