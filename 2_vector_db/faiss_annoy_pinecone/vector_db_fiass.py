import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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
        "FAISS supports both CPU and GPU operations."
    ]

    # Convert texts to vectors
    embeddings = model.encode(texts)
    print(f"Converted {len(texts)} texts to embeddings.")

    # Define vector dimension based on the model's output
    dimension = embeddings.shape[1]
    print(f"Vector dimension: {dimension}")

    # Create a FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to the index
    index.add(embeddings)
    print(f"Created FAISS index and added {len(embeddings)} vectors.")

    # Example query
    query_text = "What is FAISS?"
    query_vector = model.encode([query_text])[0]
    print(f"Encoded query: '{query_text}'")

    # Perform the query
    k = 3  # number of nearest neighbors to retrieve
    distances, indices = index.search(np.array([query_vector]), k)

    # Process and print results
    print("\nQuery Results:")
    for i in range(len(indices[0])):
        print(f"Rank: {i+1}")
        print(f"Text: {texts[indices[0][i]]}")
        print(f"Distance: {distances[0][i]}")
        print()

except Exception as e:
    print(f"An error occurred: {e}")