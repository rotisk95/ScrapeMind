import asyncio
from pyppeteer import launch
from urllib.parse import urlencode, urlunparse
import logging
import random
import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebScraper:
    def __init__(self):
        self.visited_urls = set()  # Initialize an empty set to store visited URLs


    async def search_websites(self, query, alternative_search_url=None):
        browser = await launch()
        page = await browser.newPage()
        params = urlencode({"q": query})
        search_url = urlunparse(("https", "www.google.com", "/search", "", params, ""))
        await page.goto(search_url)
        urls = await page.evaluate('''() => {
            return Array.from(document.querySelectorAll('a'))
                        .map(a => a.href)
                        .filter(url => url.startsWith('http'));
        }''')
        await browser.close()
        return urls

    async def scrape_random_website(self, query, max_retries=6):
        retries = 0
        metadata = {}
        while retries < max_retries:
            try:
                urls = await self.search_websites(query)
                search_results = [url for url in urls if not url.endswith(('.jpg', '.png', '.gif'))]
                random_url = random.choice(search_results)
                
                self.visited_urls.add(random_url)  # Add the URL to the visited set
       
                
                # Initialize metadata
                metadata['source_url'] = random_url
                metadata['scraping_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Initialize a headless browser
                browser = await launch()
                page = await browser.newPage()
                
                # Navigate to the chosen URL
                await page.goto(random_url)
                
                # Get all the text from the webpage
                text_content = await page.evaluate('''() => {
                    return document.body.innerText;
                }''')
                
                # Close the browser
                await browser.close()

                # Check if content is meaningful
                if len(text_content.split()) > 50:
                    metadata['text_length'] = len(text_content)
                    return metadata, text_content
                
                logging.warning(f"Retry {retries + 1}: Content not sufficient. Retrying...")
                
            except Exception as e:
                logging.error(f"Error encountered: {e}. Retrying...")
            
            retries += 1

        logging.error("Max retries reached. Unable to fetch content.")
        return metadata, None

    async def example_usage(self, query_word):
        metadata, content = await self.scrape_random_website(query_word)
        if content:
            print(f"Returning from scrape_random_website: {content[:250]}")
        return metadata, content

# Event loop to run async functions
if __name__ == "__main__":
    query_word = 'hello'  # Example query word from configuration
    scraper = WebScraper()
    asyncio.get_event_loop().run_until_complete(scraper.example_usage(query_word))
