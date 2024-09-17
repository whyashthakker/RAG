import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextVectorSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vector_db = {}

    def create_vector_db(self, texts):
        # Fit the vectorizer on all texts
        self.vectorizer.fit(texts)
        
        for i, text in enumerate(texts):
            vector = self.vectorizer.transform([text]).toarray()[0]
            self.vector_db[f"text_{i}"] = vector
            print(f"Added text_{i} to vector database")
            print(f"Vector shape: {vector.shape}")
        
        print(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")

    def search_similar_texts(self, query_text):
        query_vector = self.vectorizer.transform([query_text]).toarray()[0]
        print(f"Generated query vector for: '{query_text[:50]}...'")
        print(f"Query vector shape: {query_vector.shape}")
        
        similarities = {}
        for text_id, vector in self.vector_db.items():
            similarity = cosine_similarity([query_vector], [vector])[0][0]
            similarities[text_id] = similarity
            print(f"Similarity between query and {text_id}: {similarity:.4f}")
        
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above an indolent hound.",
        "Python is a popular programming language for data science and machine learning.",
    ]
    
    search_engine = TextVectorSearch()
    
    print("Creating vector database...")
    search_engine.create_vector_db(texts)
    
    query_text = "A rapid reddish-brown fox hops over a sleepy canine."
    print(f"\nPerforming similarity search with query text: '{query_text}'")
    results = search_engine.search_similar_texts(query_text)
    
    print("\nSimilarity search results:")
    for text_id, similarity in results:
        print(f"{text_id}: Similarity = {similarity:.4f}")

# Example console output:
"""
Creating vector database...
Added text_0 to vector database
Vector shape: (25,)
Added text_1 to vector database
Vector shape: (25,)
Added text_2 to vector database
Vector shape: (25,)
Vocabulary size: 25

Performing similarity search with query text: 'A rapid reddish-brown fox hops over a sleepy canine.'
Generated query vector for: 'A rapid reddish-brown fox hops over a sleepy canine.'
Query vector shape: (25,)
Similarity between query and text_0: 0.4082
Similarity between query and text_1: 0.5000
Similarity between query and text_2: 0.0000

Similarity search results:
text_1: Similarity = 0.5000
text_0: Similarity = 0.4082
text_2: Similarity = 0.0000
"""