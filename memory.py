from collections import deque
import pickle
import logging
import numpy as np  # For numerical operations
from config import DEFAULT_MEMORY_CAPACITY, LOG_FILE_PATH
from stn import SpikingTransformerNetwork  # Importing configurations
import torch
import torch.nn as nn
from collections import Counter
import spacy
from statistics import mean

class Memory:
    def __init__(self, stn_instance, capacity=DEFAULT_MEMORY_CAPACITY):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.metadata_memory = deque(maxlen=capacity)
        self.attention = {}
        self.file_path = 'memory.json'
        self.spiking_transformer_context = stn_instance

    def decay_attention(self, decay_factor=0.9):
        for info, level in self.attention.items():
            new_level = max(1, level * decay_factor)
            self.update_attention(info, new_level)

    def remember(self, feature_vector, metadata, attention_level=1):
        try:
            self.memory.append(feature_vector)
            metadata_with_attention = metadata.copy()
            metadata_with_attention['attention_level'] = attention_level
            self.metadata_memory.append(metadata_with_attention)
            self.attention[feature_vector.tobytes()] = attention_level
        except Exception as e:
            logging.error(f"Error in remember: {e}")

    def update_frequency(self, new_info):
        # Count the occurrences of each piece of information
        info_counts = Counter(self.memory)
        
        # Update attention based on frequency
        if new_info in info_counts:
            self.attention[new_info] += info_counts[new_info]

    def update_content_importance(self, new_info):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(new_info)
        num_entities = len(doc.ents)
        
        # Increase attention level based on the number of named entities
        self.attention[new_info] += num_entities


    def recall(self, filter_func=None):
        try:
            if filter_func:
                return list(filter(filter_func, self.memory))
            return list(self.memory)
        except Exception as e:
            logging.error(f"Error in recall: {e}")
            return []

    def prioritize(self):
        try:
            print("Attention Dictionary:", self.attention)  # Debugging Step 1

            if all(level <= 1 for level in self.attention.values()):  # Debugging Step 2
                print("All attention levels are less than or equal to 1.")

            prioritized_information = [info for info, level in self.attention.items() if level > 1]

            if not prioritized_information:  # Check if the list is empty
                print("No information has been prioritized.")

            return prioritized_information

        except Exception as e:
            logging.error(f"Error in prioritize: {e}")
            return []


    def update_attention(self, information, new_level):
        try:
            # Update the attention level in the self.attention dictionary
            self.attention[information] = new_level
    
            # Update the attention level in the metadata_memory deque
            for meta in self.metadata_memory:
                if meta['text_content'] == information:  # Assuming 'text_content' holds the actual information
                    meta['attention_level'] = new_level
                    break  # Break once the corresponding metadata is found and updated
        except Exception as e:
            logging.error(f"Error in update_attention: {e}")


    def save_memory(self, file_path):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.memory, f)
        except Exception as e:
            logging.error(f"Error in save_memory: {e}")

    def load_memory(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                self.memory = pickle.load(f)
        except Exception as e:
            logging.error(f"Error in load_memory: {e}")

    def update_STN_context(self):
        try:
            prioritized_information = self.prioritize()

            print("Prioritized Information:", prioritized_information)

            if len(prioritized_information) == 0:
                logging.error("Prioritized information is empty. Skipping tensor conversion.")
                return
            else:
                data_vector = torch.tensor(prioritized_information, dtype=torch.float32)

       
            self.spiking_transformer_context.update_context(data_vector)
        except Exception as e:
            logging.error(f"Error in update_STN_context: {e}")

    def retrieve_latest_data(self):
        if self.memory:
            return self.memory[-1]
        else:
            return None

    def retrieve_latest_metadata(self):
        if self.metadata_memory:
            return self.metadata_memory[-1]
        else:
            return None
    
    def retrieve_STN_context(self):
        return self.spiking_transformer_context

if __name__ == "__main__":
    logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO)
    memory = Memory()
    memory.remember(np.array([1, 2, 3]), attention_level=2)
    memory.remember(np.array([4, 5, 6]))
    print(memory.recall())
    print(memory.prioritize())
    memory.update_STN_context()
    print(f"STN Context: {memory.spiking_transformer_context.context}")