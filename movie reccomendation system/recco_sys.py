import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
from gensim.similarities import SparseTermSimilarityMatrix, SoftCosineSimilarity
from gensim.similarities.termsim import WordEmbeddingSimilarityIndex

# Function defintion to get read csv to dataframe
def get_data():
    # Read data file. Use 'id' as index and certain selected columns 
    df = pd.read_csv('movies_normalized.csv')
    
    # Return dataframe
    return df

# Function defintion to create movie titles list
def movie_list():
    # Function to create images list
    movies = make_list('title')

    #  Return list of movies
    return movies

# Function defintion to create movie images list
def images_list():
    # Function to create images list
    images = make_list('poster_path')

    # prepended image url to images
    url = "https://image.tmdb.org/t/p/original/"
    images = [url + i for i in images]
    
    #  Return list of movies posters url(partial)
    return images

# Function defintion to create a list from dataframe column
def make_list(column):
    # Function call to get dataframe
    df = get_data()

    # Create list
    series = df[column].values
    
    # Return list
    return list(series)

# Function call to get dataframe
df = get_data()

# Get normalized text from dataframe into list
documents = df['norm_text'].tolist()

# split text for each item in the list 
documents = [document.split() for document in documents]

# Create dictionary/bag-of-words of split words 
dictionary = Dictionary(documents)

# Create TF-IDF for the bag-of-words
tfidf = TfidfModel(dictionary=dictionary)

# Load saved Gensim model
w2v_model = Word2Vec.load('movie_sim_model',mmap='r')

# Get word emeddings similarity indexes
similarity_index = WordEmbeddingSimilarityIndex(w2v_model.wv)

# Create similarity (sparse) matrix
similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf, nonzero_limit=100)

# Function defintion to get top movies by selected movie
def movie_recommendations(movie, movies=movie_list(), images=images_list(), df=df, docs=documents):
    
    # Get the index of the selected movie
    movie_index = [m.lower() for m in movies].index(movie)

    # Split the contents of the movie text
    movie_text = df['norm_text'][movie_index].split()
    
    # Compute soft cosine measure between the title and the documents
    query = tfidf[dictionary.doc2bow(movie_text)]
    
    # Compute soft cosine similarities
    cos_sim = SoftCosineSimilarity(tfidf[[dictionary.doc2bow(doc) for doc in docs]], similarity_matrix)
    
    # Get similarities of the title 
    similarities = cos_sim[query]
    
    # Sort the scores to get the top similar movie indexes
    similar_movie_indexes = np.argsort(-similarities)
    
    # Creaet list of top movies and images by simiarity index of titles list
    similar_movies_titles = [movies[i] for i in similar_movie_indexes]
    similar_movie_images = [images[i] for i in similar_movie_indexes]
    
    # Create return tuples
    tiles_and_images = tuple(zip(similar_movies_titles, similar_movie_images))
    indices_and_images = tuple(zip(similar_movie_indexes, similar_movie_images))

    # Return just the titles and images
    return tiles_and_images, indices_and_images