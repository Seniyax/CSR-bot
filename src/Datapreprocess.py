import pandas as pd
import spacy
import re
from loguru import logger
import json
from nltk.tokenize import word_tokenize
import nltk


nltk.download('punkt')


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unused components for speed

# Configure logging
logger.add("logs/chatbot.log", rotation="1 MB")

def clean_text(text):
    """Clean tweet text by removing URLs, mentions, emojis, and normalizing."""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove mentions (@username)
    text = re.sub(r"@\w+", "", text)
    # Remove emojis (basic regex for common emojis)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    # Normalize whitespace and lowercase
    text = " ".join(text.lower().strip().split())
    # Use spaCy for further cleaning
    doc = nlp(text)
    cleaned = " ".join(token.text for token in doc if not token.is_punct)
    return cleaned

def chunk_text(text, max_tokens=300):
    """Split text into chunks of approximately max_tokens tokens."""
    tokens = word_tokenize(text)
    chunks = []
    current_chunk = []
    current_count = 0
    
    for token in tokens:
        current_chunk.append(token)
        current_count += 1
        if current_count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_count = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def preprocess_dataset(input_path, output_path, max_pairs=100):
    """Preprocess Kaggle dataset to extract tech support Q-A pairs."""
    logger.info("Starting data preprocessing")
    
    # Load dataset in chunks to manage memory
    chunksize = 500
    qa_pairs = []
    
    # Tech support company handles
    tech_companies = ["AppleSupport", "MicrosoftHelps", "Google", "AmazonHelp"]
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        # Filter customer queries (no in_response_to_tweet_id)
        queries = chunk[chunk["in_response_to_tweet_id"].isna() & 
                       chunk["text"].str.contains("help|issue|problem|support", case=False, na=False)]
        
        for _, query in queries.iterrows():
            # Find company reply
            reply = chunk[(chunk["in_response_to_tweet_id"] == query["tweet_id"]) & 
                         chunk["author_id"].isin(tech_companies)]
            
            if not reply.empty:
                question = clean_text(query["text"])
                answer = clean_text(reply.iloc[0]["text"])
                if question and answer:  # Ensure non-empty
                    qa_text = f"Question: {question} Answer: {answer}"
                    # Chunk the Q-A pair
                    chunks = chunk_text(qa_text)
                    for i, chunk in enumerate(chunks):
                        qa_pairs.append({
                            "id": f"{query['tweet_id']}_{i}",
                            "text": chunk,
                            "metadata": {
                                "company": reply.iloc[0]["author_id"],
                                "category": "tech_support"
                            }
                        })
        
        # Stop after collecting max_pairs
        if len(qa_pairs) >= max_pairs:
            break
    
    # Save processed data
    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    
    logger.info(f"Processed {len(qa_pairs)} Q-A pairs, saved to {output_path}")
    return qa_pairs

if __name__ == "__main__":
    input_path = "data/twcs.csv"
    output_path = "data/processed/faqs_processed.json"
    preprocess_dataset(input_path, output_path, max_pairs=100)