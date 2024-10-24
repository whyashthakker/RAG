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
    # Word2Vec is a popular method for generating word embeddings
    # It learns vector representations of words that capture semantic relationships
    sentences = [['cat', 'say', 'meow'], ['dog', 'say', 'woof']]
    model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=4)
    # Parameters:
    # - vector_size=10: Dimensionality of the word vectors
    # - window=5: Maximum distance between current and predicted word within a sentence
    # - min_count=1: Ignores all words with total frequency lower than this
    # - workers=4: Number of CPU cores to use for training
    print("Word Embedding for 'cat':", model.wv['cat'])

# Sentence Embeddings
def sentence_embeddings():
    # SentenceTransformer is a library for state-of-the-art sentence embeddings
    # It's based on BERT architecture and fine-tuned for generating sentence embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # 'paraphrase-MiniLM-L6-v2' is the name of the pre-trained model being used
    sentences = ["This is an example sentence", "Each sentence is converted to a vector"]
    embeddings = model.encode(sentences)
    print("Sentence Embedding shape:", embeddings.shape)
    print("First sentence embedding:", embeddings[0][:5])  # First 5 dimensions

# Image Embeddings
def image_embeddings():
    # ResNet18 is a convolutional neural network architecture
    # It's designed to handle the vanishing gradient problem in deep networks
    model = models.resnet18(pretrained=True)
    # pretrained=True: Uses weights pre-trained on ImageNet
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last fully connected layer
    model.eval()
    
    # Image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open("2_vector_db/1.jpg")  
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(img_tensor).squeeze()
    
    print("Image Embedding shape:", embedding.shape)
    print("First few dimensions of image embedding:", embedding[:5])

# Graph Embeddings
def graph_embeddings():
    # Node2Vec is an algorithmic framework for learning continuous feature representations for nodes in networks
    G = karate_club_graph()
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    # Parameters:
    # - dimensions=64: Dimensionality of the node embeddings
    # - walk_length=30: Length of walk per source
    # - num_walks=200: Number of walks per source
    # - workers=4: Number of CPU cores to use
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Additional parameters for fit:
    # - window=10: Maximum distance between the current and predicted node in the random walk
    # - min_count=1: Ignores all nodes with total frequency lower than this
    # - batch_words=4: Number of words to be processed in a single batch
    
    print("Graph Embedding for node 0:", model.wv['0'])

# Audio Embeddings
def audio_embeddings():
    # This uses Mel-frequency cepstral coefficients (MFCCs)
    # MFCCs are commonly used features in speech and audio processing
    audio_path = "2_vector_db/1.wav"  # Replace with actual audio path
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Parameters:
    # - y: The input audio time series
    # - sr: The sampling rate of y
    # - n_mfcc=13: The number of MFCCs to return
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