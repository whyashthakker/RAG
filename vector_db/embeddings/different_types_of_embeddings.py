import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from networkx import karate_club_graph, spring_layout
from node2vec import Node2Vec
import librosa

# Word Embeddings
def word_embeddings():
    sentences = [['cat', 'say', 'meow'], ['dog', 'say', 'woof']]
    model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=4)
    print("Word Embedding for 'cat':", model.wv['cat'])

# Sentence Embeddings
def sentence_embeddings():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentences = ["This is an example sentence", "Each sentence is converted to a vector"]
    embeddings = model.encode(sentences)
    print("Sentence Embedding shape:", embeddings.shape)
    print("First sentence embedding:", embeddings[0][:5])  # First 5 dimensions

# Image Embeddings
def image_embeddings():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last fully connected layer
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open("vector_db/1.jpg")  # Replace with actual image path
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(img_tensor).squeeze()
    
    print("Image Embedding shape:", embedding.shape)
    print("First few dimensions of image embedding:", embedding[:5])

# Graph Embeddings
def graph_embeddings():
    G = karate_club_graph()
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    print("Graph Embedding for node 0:", model.wv['0'])

# Audio Embeddings
def audio_embeddings():
    audio_path = "vector_db/1.wav"  # Replace with actual audio path
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_embedding = np.mean(mfccs.T, axis=0)
    
    print("Audio Embedding (MFCC):", mfcc_embedding)

if __name__ == "__main__":
    print("Word Embeddings:")
    word_embeddings()
    
    print("\nSentence Embeddings:")
    sentence_embeddings()
    
    print("\nImage Embeddings:")
    image_embeddings()
    
    print("\nGraph Embeddings:")
    graph_embeddings()
    
    print("\nAudio Embeddings:")
    audio_embeddings()