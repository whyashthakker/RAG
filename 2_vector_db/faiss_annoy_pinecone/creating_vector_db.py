from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

try:
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    print("Pinecone initialized successfully.")

    # Load a pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model loaded.")

    # Sample custom data (text)
    texts = [
        "Pinecone is a vector database.",
        "Vectors represent data in numerical form.",
        "Embedding models convert text to vectors."
    ]

    # Convert texts to vectors
    embeddings = model.encode(texts)
    print(f"Converted {len(texts)} texts to embeddings.")

    # Define vector dimension based on the model's output
    dimension = embeddings.shape[1]
    print(f"Vector dimension: {dimension}")

    # Define index name
    index_name = "ragudemy"

    # Check if index already exists
    existing_indexes = pc.list_indexes()
    if index_name not in existing_indexes:
        # Create the index
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new index: {index_name}")
    else:
        print(f"Index {index_name} already exists.")

    # Connect to the index
    index = pc.Index(index_name)
    print(f"Connected to index: {index_name}")

    # Prepare the data for insertion (ID and vector pairs)
    vector_data = [(str(i), embeddings[i].tolist(), {"text": texts[i]}) for i in range(len(embeddings))]

    # Insert vectors into Pinecone
    index.upsert(vectors=vector_data)
    print(f"Inserted {len(vector_data)} vectors into the index.")

    # Example query
    query_text = "How does Pinecone work?"
    query_vector = model.encode([query_text])[0].tolist()
    print(f"Encoded query: '{query_text}'")

    # Perform the query
    result = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    # Process and print results
    if result['matches']:
        print("\nQuery Results:")
        for match in result['matches']:
            print(f"ID: {match['id']}, Score: {match['score']}")
            print(f"Text: {match['metadata']['text']}")
            print()
    else:
        print("No matches found in the result.")

except Exception as e:
    print(f"An error occurred: {e}")