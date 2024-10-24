import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

def get_image_embedding(image_path):
    # Open the image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    print(f"Opened image: {image_path}")
    
    # Resize to a standard size
    image = image.resize((224, 224))
    print(f"Resized image to 224x224")
    
    # Convert to numpy array and normalize pixel values
    image = np.array(image) / 255.0
    print(f"Converted image to numpy array and normalized")
    
    # Flatten the image into a 1D vector
    vector = image.flatten()
    print(f"Flattened image to vector of shape: {vector.shape}")
    
    return vector

def create_vector_db(image_files):
    vector_db = {}
    for image_file in image_files:
        vector_db[image_file] = get_image_embedding(image_file)
        print(f"Added {image_file} to vector database")
    return vector_db

def search_similar_images(query_image_path, vector_db):
    query_vector = get_image_embedding(query_image_path)
    print(f"Generated query vector for: {query_image_path}")
    
    similarities = {}
    for image_file, vector in vector_db.items():
        similarity = cosine_similarity([query_vector], [vector])[0][0]
        similarities[image_file] = similarity
        print(f"Similarity between query and {image_file}: {similarity:.4f}")
    
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

if __name__ == "__main__":
    image_files = ["2_vector_db/1.jpg", "2_vector_db/2.png"]
    print("Creating vector database...")
    vector_db = create_vector_db(image_files)
    
    query_image_path = "2_vector_db/3.png"
    print(f"\nPerforming similarity search with query image: {query_image_path}")
    results = search_similar_images(query_image_path, vector_db)
    
    print("\nSimilarity search results:")
    for image_file, similarity in results:
        print(f"{image_file}: Similarity = {similarity:.4f}")
