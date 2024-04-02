import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import zipfile

# Base URL and scheme for constructing full URLs
base_url = 'https://www.dafont.com'
scheme = 'https:'

# Function to extract font page URLs from a specific listing page
def get_font_page_urls(listing_url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(listing_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    font_links = soup.select('a.dl')  # Select download links directly
    font_page_urls = [urljoin(scheme, link['href']) for link in font_links if 'href' in link.attrs]
    return font_page_urls

def download_and_extract_zip(url, save_path, extract_to):
    # Make the request
    response = requests.get(url)
    
    # Ensure the request was successful
    if response.status_code == 200:
        # Save the zip file to the filesystem
        with open(save_path, 'wb') as file:
            file.write(response.content)
        
        # Extract the zip file
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Zip content extracted to '{extract_to}'")
        import os
        os.remove(save_path)
    else:
        print(f"Failed to download the zip file. Status code: {response.status_code}")


# Main function to orchestrate the scraping across multiple pages
def main():
    # Define the range of pages you want to scrape
    start_page = 1
    end_page = 2  # Adjust this to the last page you want to scrape
    
    for page in range(start_page, end_page + 1):
        listing_url = f'https://www.dafont.com/new.php?page={page}'
        print(f"Processing {listing_url}")
        font_page_urls = get_font_page_urls(listing_url)
        
        for url in font_page_urls:
            download_link = urljoin(scheme, url)  # The URL is already the download link
            print(f"Download Link: {download_link}")
            # Here you could add your logic to download the font files
            download_and_extract_zip(download_link, f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/templates/fonts/temp', f'/Users/arnaudstiegler/llm-table-extraction/synth_data_gen/templates/fonts/{url.split("?f=")[-1]}')

if __name__ == "__main__":
    main()