import asyncio
import json
import logging
import numpy as np
import torch
from web_scraper import WebScraper
from content_processor import ContentProcessor
from memory import Memory
from stn import SpikingTransformerNetwork
from config import WEB_SCRAPING_GENERAL_URLS, WEB_SCRAPING_SPECIFIC_URLS

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize TRAINING_MODE
TRAINING_MODE = True

# Initialize visited URLs
visited_urls = set()

def load_visited_urls():
    try:
        with open('visited_urls.json', 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        logging.info("No visited_urls.json found.")
        return set()

def save_visited_urls(urls):
    with open('visited_urls.json', 'w') as f:
        json.dump(list(urls), f)

async def process_url(url, web_scraper, content_processor, stn_context, memory_module):
    logging.info(f"Processing URL: {url}")
    metadata, scraped_data = await web_scraper.example_usage(url)
    if not scraped_data:
        logging.warning("No data scraped.")
        return

    processed_data = content_processor.process(scraped_data, metadata)
    features = stn_context.extract_features(processed_data)
    encoded_data = stn_context.encode_features(features)
    stn_context.update_context(encoded_data)
    output_data = stn_context(encoded_data)
    memory_module.remember(output_data.detach().numpy(), metadata, 1)
    memory_module.update_STN_context()

async def inference_mode(memory_module):
    user_input = input("Enter your question for inference: ").strip()
    prioritized_data = memory_module.prioritize()

    if prioritized_data:
        # Generate a response based on prioritized data
        response = "Answer based on " + str(prioritized_data)
    else:
        response = "I don't have enough information to answer that."
    
    print(f"Model's response: {response}")

async def main():
    global TRAINING_MODE
    visited_urls = load_visited_urls()

    web_scraper = WebScraper()
    content_processor = ContentProcessor()
    stn = SpikingTransformerNetwork(input_dim=544, hidden_dim=1024, output_dim=2048)
    memory_module = Memory(stn)

    while True:
        try:
            if TRAINING_MODE:
                for url in WEB_SCRAPING_GENERAL_URLS + WEB_SCRAPING_SPECIFIC_URLS:
                    if url in visited_urls:
                        logging.info(f"URL {url} has already been visited.")
                        continue

                    try:
                        stn_context = memory_module.retrieve_STN_context()
                        await process_url(url, web_scraper, content_processor, stn_context, memory_module)
                        visited_urls.add(url)
                    except Exception as e:
                        logging.error(f"Error processing URL {url}: {e}")
                        continue
                
                save_visited_urls(visited_urls)

        except KeyboardInterrupt:
            action = input("What would you like to do next? (T)rain / (I)nference / (E)xit: ").strip().lower()
            if action == 't':
                TRAINING_MODE = True
            elif action == 'i':
                TRAINING_MODE = False
            elif action == 'e':
                break
            else:
                print("Invalid option.")

        else:
            if not TRAINING_MODE:
                await inference_mode(memory_module)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

    
'''
            if not use_distilbert:
                # Append the raw scraped data to the buffer
                raw_scraped_data_buffer.append(scraped_data)
                buffer_length = len(raw_scraped_data_buffer)
            else:
                # Add the new feature vector to the buffer
                feature_buffer.append(output_data.detach().numpy())
                buffer_length = len(feature_buffer)
                
            metadata_list.append(metadata)
            # Move decay_attention() inside the condition
            if buffer_length >= MIN_SAMPLES_REQUIRED_FOR_CLUSTERING:
                if iteration_counter % decay_interval == 0:
                    memory_module.decay_attention()
                    print("Decayed attention levels.")
                
                
                if use_distilbert:
                    # Convert feature_data to numpy if it's a tensor
                    if isinstance(feature_buffer, torch.Tensor):
                        feature_data_np = feature_buffer.detach().numpy()
                    else:
                        feature_data_np = np.array(feature_buffer)
                    cluster_results, clusters_metadata = cluster_analyzer.analyze_clusters(raw_text_data=None,feature_data_np=feature_data_np)
                    # Clear the buffer
                    feature_buffer.clear()                
                else:    
                    cluster_results, clusters_metadata = cluster_analyzer.analyze_clusters(raw_text_data=raw_scraped_data_buffer,feature_data_np=None)
                    # Clear the buffer
                    raw_scraped_data_buffer.clear() 
                # Check if clustering was successful
                if not cluster_results:
                    print("Clustering returned empty results. Skipping this iteration.")
                    continue
                '''
                # metadata_list.append(clusters_metadata)
                
'''
           else:
              print("Insufficient samples for clustering. Skipping decay and STN update.")
'''


    
        
