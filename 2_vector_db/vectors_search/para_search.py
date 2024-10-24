import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

class LongTextVectorSearch:
    def __init__(self, chunk_size=100):
        self.vectorizer = TfidfVectorizer()
        self.vector_db = []
        self.chunk_size = chunk_size
        self.chunk_texts = []

    def split_text(self, text):
        words = text.split()
        return [' '.join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    def create_vector_db(self, texts):
        all_chunks = []
        for i, text in enumerate(texts):
            chunks = self.split_text(text)
            all_chunks.extend(chunks)
            self.chunk_texts.extend([f"text_{i}_chunk_{j}" for j in range(len(chunks))])
            print(f"Added {len(chunks)} chunks for text_{i}")

        # Fit the vectorizer on all chunks
        self.vectorizer.fit(all_chunks)
        
        # Transform all chunks at once
        self.vector_db = self.vectorizer.transform(all_chunks)
        
        print(f"Total chunks: {len(self.chunk_texts)}")
        print(f"Vector shape: {self.vector_db.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")

    def search_similar_texts(self, query_text, top_k=5):
        query_vector = self.vectorizer.transform([query_text])
        print(f"Generated query vector for: '{query_text[:50]}...'")
        
        # Compute similarities for all chunks at once
        similarities = cosine_similarity(query_vector, self.vector_db).flatten()
        
        # Get top_k similar chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for index in top_indices:
            chunk_id = self.chunk_texts[index]
            similarity = similarities[index]
            results.append((chunk_id, similarity))
            print(f"Similarity between query and {chunk_id}: {similarity:.4f}")
        
        return results

if __name__ == "__main__":
    # Example with longer texts
    texts = [
        """What Is a Vector Database?

A vector database is a specialized data storage and retrieval system designed to handle high-dimensional vector data efficiently. In the context of machine learning and artificial intelligence, vectors are numerical representations of data objects, such as words, images, or user behaviors, encoded in a multi-dimensional space. These vectors, often called embeddings, capture the semantic or contextual meaning of the data.

Key Characteristics:

Storage of High-Dimensional Data: Capable of storing vectors with hundreds or thousands of dimensions.
Efficient Similarity Search: Optimized for operations like nearest neighbor search, which finds vectors similar to a given query vector.
Scalability: Designed to handle large volumes of data, potentially billions of vectors.
Integration with AI Models: Works seamlessly with machine learning models that generate embeddings.
Differences Between Vector Databases and Traditional Databases

Data Representation:

Traditional Databases: Store structured data (tables with rows and columns) or unstructured data (documents).
Vector Databases: Store data as high-dimensional vectors representing complex data relationships.
Query Mechanisms:

Traditional Databases: Use SQL queries or key-value lookups for exact matches or range queries.
Vector Databases: Perform similarity searches using distance metrics (e.g., cosine similarity, Euclidean distance).
Indexing Structures:

Traditional Databases: Use B-trees, hash indexes, or inverted indexes.
Vector Databases: Employ specialized indexes like k-d trees, VP-trees, or approximate nearest neighbor algorithms.
Performance Optimization:

Traditional Databases: Optimize for transaction throughput and ACID properties.
Vector Databases: Optimize for low-latency similarity searches over large datasets.
Importance and Applications of Vector Databases

Handling Complex Data: Enable the storage and retrieval of data types that are difficult to manage with traditional databases (e.g., images, audio, text semantics).
Real-Time Recommendations: Provide immediate suggestions based on user interactions by quickly finding similar items.
Enhanced Search Capabilities: Allow semantic search, where queries return results based on meaning rather than keyword matching.
Machine Learning Integration: Serve as a backend for AI applications that require fast access to vector representations.
Use Cases in Industries

E-commerce:

Product Recommendations: Suggest products similar to those a user has viewed or purchased.
Visual Search: Allow users to search for products using images.
Media and Entertainment:

Content Recommendations: Suggest movies, songs, or articles based on user preferences.
Image and Video Retrieval: Find visually similar media content.
Healthcare:

Medical Imaging: Compare patient scans to identify anomalies or similar cases.
Genomics: Analyze genetic data represented as vectors.
Finance:

Fraud Detection: Identify unusual transactions by comparing transaction vectors.
Risk Assessment: Evaluate investment similarities and portfolio diversification."""  # ~650 words
    ]
    
    search_engine = LongTextVectorSearch(chunk_size=50)  # 50 words per chunk
    
    print("Creating vector database...")
    search_engine.create_vector_db(texts)
    
    query_text = "What is a Pokemon?"
    print(f"\nPerforming similarity search with query text: '{query_text}'")
    results = search_engine.search_similar_texts(query_text)
    
    print("\nTop 5 similar chunks:")
    for chunk_id, similarity in results:
        print(f"{chunk_id}: Similarity = {similarity:.4f}")