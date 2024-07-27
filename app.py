import streamlit as st
from neo4j import GraphDatabase
from bs4 import BeautifulSoup
import requests
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from transformers import pipeline
from textblob import TextBlob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

# Initialize Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class ContentEvaluator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def evaluate_response(self, query, response):
        # Vectorize the query and response
        vectors = self.vectorizer.fit_transform([query, response])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        
        # Extract key sentences
        sentences = sent_tokenize(response)
        relevant_sentences = [sent for sent in sentences if any(word in sent.lower() for word in query.lower().split())]
        
        # Extract named entities
        doc = nlp(response)
        named_entities = [ent.text for ent in doc.ents]
        
        return similarity, relevant_sentences, named_entities

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.evaluator = ContentEvaluator()

    def get_exact_document(self, query):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Document {query: $query}) RETURN d.content AS content",
                {"query": query}
            )
            record = result.single()
            return self.preprocess_content(record["content"]) if record else None

    def store_document(self, query, content):
        preprocessed_content = self.preprocess_content(content)
        with self.driver.session() as session:
            session.run(
                "MERGE (d:Document {query: $query}) "
                "SET d.content = $content",
                {"query": query, "content": preprocessed_content}
            )

    def close(self):
        if self.driver:
            self.driver.close()

    @staticmethod
    def scrape_website(url, query):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find the main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            
            if main_content:
                paragraphs = main_content.find_all('p')
            else:
                paragraphs = soup.find_all('p')
            
            content = ' '.join([p.get_text() for p in paragraphs])
            
            # Basic content filtering
            if len(content.split()) < 50:  # Ignore very short content
                return ""
            
            return Neo4jClient.preprocess_content(content)
        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""

    @staticmethod
    def search_and_scrape(query, search_engine):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        if search_engine == "google":
            search_url = f"https://www.google.com/search?q={query}"
        elif search_engine == "bing":
            search_url = f"https://www.bing.com/search?q={query}"
        elif search_engine == "duckduckgo":
            search_url = f"https://html.duckduckgo.com/html/?q={query}"
        else:
            return ""

        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            if search_engine == "google":
                results = soup.find_all('div', class_='g')
            elif search_engine == "bing":
                results = soup.find_all('li', class_='b_algo')
            else:  # DuckDuckGo
                results = soup.find_all('div', class_='result')

            all_content = []
            for result in results[:5]:  # Check top 5 results
                link = result.find('a', href=True)
                if link and link['href'].startswith('http'):
                    content = Neo4jClient.scrape_website(link['href'], query)
                    if content:
                        all_content.append(content)
                        logger.info(f"Successfully scraped content from {link['href']} using {search_engine}")
            
            return ' '.join(all_content)
        except requests.RequestException as e:
            logger.error(f"Error occurred while scraping {search_engine}: {e}")
            return ""

    def multi_source_scrape(self, query):
        search_engines = ["google", "bing", "duckduckgo"]
        all_content = []

        with ThreadPoolExecutor(max_workers=len(search_engines)) as executor:
            future_to_engine = {executor.submit(self.search_and_scrape, query, engine): engine for engine in search_engines}
            for future in as_completed(future_to_engine):
                engine = future_to_engine[future]
                try:
                    content = future.result()
                    if content:
                        score = self.evaluator.evaluate_response(query, content)[0]
                        all_content.append((engine, content, score))
                except Exception as exc:
                    logger.error(f"{engine} generated an exception: {exc}")

        return all_content

    @staticmethod
    def preprocess_content(content):
        # Remove special characters and extra whitespace
        content = re.sub(r'[^\w\s]', '', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Tokenize and lemmatize
        tokens = word_tokenize(content.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        return ' '.join(lemmatized_tokens)

    @staticmethod
    def format_response(content, query, similarity, relevant_sentences, named_entities):
        if not content or len(content.split()) < 10:
            return f"I'm sorry, but I couldn't find enough information about '{query}'. Could you try rephrasing your question or asking about something else?"

        try:
            # Use relevant sentences to create a more focused summary
            relevant_content = " ".join(relevant_sentences)
            summary = summarizer(relevant_content, max_length=200, min_length=100, do_sample=False)[0]['summary_text']
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            summary = " ".join(relevant_sentences[:3])  # Use top 3 relevant sentences if summarization fails

        # Create a narrative response
        response = f"{query.capitalize()} refers to {summary} "
        if named_entities:
            response += f"Key elements associated with this topic include {', '.join(named_entities[:3])}. "
        response += f"This information is based on sources with a relevance score of {similarity:.2f}."

        return response

def post_process(response):
    # Use TextBlob for grammar correction
    blob = TextBlob(response)
    corrected_response = str(blob.correct())
    
    # Ensure the response is not too short
    if len(corrected_response.split()) < 20:
        corrected_response += " However, the available information seems limited. You may want to try rephrasing your query for more comprehensive results."
    
    return corrected_response

def main():
    st.title("RAG-based Query-Answer System")
    
    # Use Streamlit secrets for credentials
    neo4j_uri = st.secrets["NEO4J_URI"]
    neo4j_user = st.secrets["NEO4J_USER"]
    neo4j_password = st.secrets["NEO4J_PASSWORD"]

    neo4j_client = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)

    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("Retrieving and processing information..."):
            try:
                stored_response = neo4j_client.get_exact_document(user_query)
                evaluator = ContentEvaluator()

                if stored_response:
                    similarity, relevant_sentences, named_entities = evaluator.evaluate_response(user_query, stored_response)
                    formatted_response = Neo4jClient.format_response(stored_response, user_query, similarity, relevant_sentences, named_entities)
                    final_response = post_process(formatted_response)
                    st.write("Answer (from database):")
                    st.write(final_response)
                    logger.info(f"Retrieved and processed answer from database for query: {user_query}")
                else:
                    logger.info(f"No stored response found for query: {user_query}. Attempting multi-source web scraping.")
                    web_responses = neo4j_client.multi_source_scrape(user_query)
                    
                    if web_responses:
                        best_response = max(web_responses, key=lambda x: x[2])
                        best_engine, best_content, best_score = best_response
                        
                        neo4j_client.store_document(user_query, best_content)
                        similarity, relevant_sentences, named_entities = evaluator.evaluate_response(user_query, best_content)
                        formatted_response = Neo4jClient.format_response(best_content, user_query, similarity, relevant_sentences, named_entities)
                        final_response = post_process(formatted_response)
                        
                        st.write(f"Answer (from {best_engine}):")
                        st.write(final_response)
                        logger.info(f"Retrieved, processed, and presented best answer from {best_engine} for query: {user_query}")
                        
                        # Log detailed information for debugging
                        logger.debug(f"Query: {user_query}")
                        logger.debug(f"Best content: {best_content[:500]}...")  # Log first 500 characters
                        logger.debug(f"Similarity score: {similarity}")
                        logger.debug(f"Relevant sentences: {relevant_sentences}")
                        logger.debug(f"Named entities: {named_entities}")
                        logger.debug(f"Formatted response: {formatted_response}")
                        logger.debug(f"Final response: {final_response}")
                    else:
                        st.write(f"I'm sorry, but I couldn't find any information about '{user_query}'. Could you try rephrasing your question or asking about something else?")
                        logger.warning(f"No information found for query: {user_query}")

                st.write("Note: This answer is based on retrieved and processed information. While efforts have been made to ensure accuracy, please verify crucial information from authoritative sources.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error processing query '{user_query}': {str(e)}")

    neo4j_client.close()

if __name__ == "__main__":
    main()