from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import logging
from config import LOG_FILE_PATH  # Importing logging configuration
import re

# Ensure NLTK resources are downloaded
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK resources: {e}")

class ContentProcessor:
    def __init__(self, language='english'):
        self.language = language

    def preprocess_text(self, text, metadata):
        try:
            # Tokenize the text
            tokens = word_tokenize(text)
        
            # Convert to lowercase
            tokens = [word.lower() for word in tokens]
        
            # Remove punctuation
            word_tokens = [word for word in tokens if word.isalpha()]
        
            # Identify stopwords
            stop_words = set(stopwords.words(self.language))
            stop_word_tokens = [word for word in word_tokens if word in stop_words]
        
            # Identify new words
            new_word_tokens = [word for word in word_tokens if word not in stop_words]
        
            # Identify numbers
            number_tokens = [word for word in tokens if word.isdigit()]
        
            # Identify special characters
            special_character_tokens = [word for word in tokens if re.match(r"[^a-zA-Z0-9]", word)]
        
            # Update the metadata dictionary
            metadata['tokens'] = {
                "new_words": new_word_tokens,
                "stop_words": stop_word_tokens,
                "numbers": number_tokens,
                "special_characters": special_character_tokens
            }
            
            return metadata
        except Exception as e:
            logging.error(f"Error in preprocess_text: {e}")
            return {}
    
    def process(self, text, metadata):
        # Wrapper method for preprocessing and information extraction
        processed_data = self.preprocess_text(text, metadata)
        return processed_data

# Example usage
if __name__ == "__main__":
    logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO)  # Initialize logging
    processor = ContentProcessor()
    text = "This is an example text with some new words like mommy and daddy."
    metadata = {"source_url": "https://www.example.com", "scraping_time": "2023-08-26 16:27:03", "text_length": 1234}
    processed_data = processor.process(text, metadata)
    print("Processed Data:", processed_data)
