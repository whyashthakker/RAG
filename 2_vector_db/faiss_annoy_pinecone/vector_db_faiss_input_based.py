import numpy as np
import faiss
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import time

#This function creates a FAISS index using the L2 (Euclidean) distance, adds the embeddings to the index, and performs a k-nearest neighbor search for the query vector.
def faiss_search(embeddings, query_vector, k):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(np.array([query_vector]), k)
    return distances[0], indices[0]

#This function creates an Annoy index using angular (cosine) distance, adds embeddings to the index, builds the index with 10 trees, and performs a k-nearest neighbor search.
def annoy_search(embeddings, query_vector, k):
    dimension = embeddings.shape[1]
    index = AnnoyIndex(dimension, 'angular')
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(10)  # 10 trees - more trees gives higher precision when querying
    indices, distances = index.get_nns_by_vector(query_vector, k, include_distances=True)
    return distances, indices

try:
    # Load a pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded.")

    # Sample custom data (text)
    texts = [
        "FAISS is a library for efficient similarity search.",
        "Vectors represent data in numerical form.",
        "Embedding models convert text to vectors.",
        "Local vector databases can be faster for small datasets.",
        "FAISS supports both CPU and GPU operations.",
        "Annoy is an efficient library for approximate nearest neighbor search.",
        "Vector databases are crucial for large-scale machine learning applications.",
        "Similarity search is used in recommendation systems and information retrieval.",
        "Dimensionality reduction techniques can improve search efficiency.",
        "Cosine similarity is a common metric in vector space models."
    ]

    # Convert texts to vectors
    embeddings = model.encode(texts)
    print(f"Converted {len(texts)} texts to embeddings.")

    # Define vector dimension based on the model's output
    dimension = embeddings.shape[1]
    print(f"Vector dimension: {dimension}")

    # Get user input for query
    query_text = input("Enter your query: ")
    query_vector = model.encode([query_text])[0]
    print(f"Encoded query: '{query_text}'")

    # Perform the query with FAISS
    print("\nFAISS Results:")
    start_time = time.time()
    faiss_distances, faiss_indices = faiss_search(embeddings, query_vector, 3)
    faiss_time = time.time() - start_time
    for i, (distance, idx) in enumerate(zip(faiss_distances, faiss_indices)):
        print(f"Rank: {i+1}")
        print(f"Text: {texts[idx]}")
        print(f"Distance: {distance}")
        print(f"Score: {1 / (1 + distance):.4f}")  # Convert distance to a similarity score
        print()
    print(f"FAISS search time: {faiss_time:.6f} seconds")

    # Perform the query with Annoy
    print("\nAnnoy Results:")
    start_time = time.time()
    annoy_distances, annoy_indices = annoy_search(embeddings, query_vector, 3)
    annoy_time = time.time() - start_time
    for i, (distance, idx) in enumerate(zip(annoy_distances, annoy_indices)):
        print(f"Rank: {i+1}")
        print(f"Text: {texts[idx]}")
        print(f"Distance: {distance}")
        print(f"Score: {1 / (1 + distance):.4f}")  # Convert distance to a similarity score
        print()
    print(f"Annoy search time: {annoy_time:.6f} seconds")

except Exception as e:
    print(f"An error occurred: {e}")