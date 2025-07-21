import pandas as pd
import spacy
import re
from loguru import logger
import json
from nltk.tokenize import word_tokenize
import nltk


import pandas as pd
import spacy
import re
from loguru import logger
import json
from nltk.tokenize import word_tokenize
import nltk
import os

# Download NLTK data
nltk.download('punkt')

# Initialize spaCy for text processing
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Disable unused components for speed

# Configure logging
logger.add("customer_support_chatbot/logs/chatbot.log", rotation="1 MB")

def clean_text(text, tech_companies):
    """Clean tweet text, removing URLs, user mentions (except tech companies), emojis, and normalizing."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove user mentions (except tech companies)
    tech_mention_pattern = "|".join([f"@{company}" for company in tech_companies])
    text = re.sub(r"@\w+", lambda x: x.group(0) if x.group(0) in tech_mention_pattern else "", text)
    # Remove emojis (basic regex for common emojis)
    text = re.sub(r"[^\w\s.,!?@]", "", text)
    # Normalize whitespace and lowercase
    text = " ".join(text.lower().strip().split())
    # Use spaCy for further cleaning
    doc = nlp(text)
    cleaned = " ".join(token.text for token in doc if not token.is_punct)
    return cleaned

def chunk_text(text, max_tokens=300):
    """Split text into chunks of approximately max_tokens tokens."""
    if not text:
        return []
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
    
    # Tech support company handles
    tech_companies = ["AppleSupport", "MicrosoftHelps", "Google", "AmazonHelp", "DellCares", "HPSupport","AdobeCare"]
    
    # Load replies dataset
    logger.info("Loading replies dataset")
    replies_df = pd.read_csv(input_path, usecols=["tweet_id", "author_id", "text", "in_response_to_tweet_id"],
                             dtype={"tweet_id": str, "author_id": str, "in_response_to_tweet_id": str})
    replies_df.columns = replies_df.columns.str.lower().str.replace(" ", "_")
    replies_df["in_response_to_tweet_id"] = replies_df["in_response_to_tweet_id"].astype(str).replace("nan", "")
    
    # Filter tech support replies
    tech_replies = replies_df[replies_df["author_id"].isin(tech_companies)]
    logger.info(f"Found {len(tech_replies)} potential tech support replies")
    
    # Process queries in chunks
    chunksize = 500
    qa_pairs = []
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, 
                             dtype={"tweet_id": str, "in_response_to_tweet_id": str, "author_id": str}):
        chunk.columns = chunk.columns.str.lower().str.replace(" ", "_")
        chunk["tweet_id"] = chunk["tweet_id"].astype(str)
        chunk["in_response_to_tweet_id"] = chunk["in_response_to_tweet_id"].astype(str).replace("nan", "")
        
        # Filter customer queries (not from tech companies, contains tech support keywords)
        queries = chunk[(chunk["in_response_to_tweet_id"] == "") & 
                        ~chunk["author_id"].isin(tech_companies) &
                        chunk["text"].str.contains("help|issue|problem|support|error|bug|technical|fix|trouble|device|software|update", 
                                                 case=False, na=False)]
        logger.info(f"Found {len(queries)} queries in chunk")
        
        for _, query in queries.iterrows():
            # Find reply in tech_replies
            reply = tech_replies[tech_replies["in_response_to_tweet_id"] == query["tweet_id"]]
            
            if not reply.empty:
                question = clean_text(query["text"], tech_companies)
                answer = clean_text(reply.iloc[0]["text"], tech_companies)
                if question and answer:  # Ensure non-empty
                    qa_text = f"Question: {question} Answer: {answer}"
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
                    logger.debug(f"Added Q-A pair for tweet_id {query['tweet_id']}: {qa_text[:50]}...")
        
        if len(qa_pairs) >= max_pairs:
            break
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    
    logger.info(f"Processed {len(qa_pairs)} Q-A pairs, saved to {output_path}")
    return qa_pairs

if __name__ == "__main__":
    input_path = "data/twcs.csv"
    output_path = "data/faqs_processed.json"
    preprocess_dataset(input_path, output_path, max_pairs=100)