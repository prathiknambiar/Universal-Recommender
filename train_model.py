import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

movies = pd.read_csv("data/movielens/movie.csv")
ratings = pd.read_csv("data/movielens/rating.csv")
tags = pd.read_csv("data/movielens/tag.csv")

rating_counts = ratings.groupby('movieId').size()
popular_movies = rating_counts[rating_counts > 100].index

ratings = ratings[ratings['movieId'].isin(popular_movies)]
movies = movies[movies['movieId'].isin(popular_movies)]
tags = tags[tags['movieId'].isin(popular_movies)]
movie_tags = tags.groupby("movieId")["tag"].apply(
    lambda x: " ".join(x.dropna().astype(str))
).reset_index()
movies_with_tags = movies.merge(movie_tags, on="movieId", how="left")
movies_with_tags["tag"] = movies_with_tags["tag"].fillna("")

movie_matrix = ratings.pivot_table(
    index="movieId",
    columns="userId",
    values="rating"
)

tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = tfidf.fit_transform(movies["genres"])
genre_similarity = cosine_similarity(genre_matrix)
genre_indices = {movie_id: i for i, movie_id in enumerate(movies["movieId"])}
user_means = movie_matrix.mean(axis=0)
movie_matrix_centered = movie_matrix.sub(user_means, axis=1).fillna(0)

tfidf_tag = TfidfVectorizer(stop_words='english')
tag_matrix = tfidf.fit_transform(movies_with_tags["tag"])
tag_similarity = cosine_similarity(tag_matrix)
tag_indices = {movie_id: i for i, movie_id in enumerate(movies_with_tags["movieId"])}

movie_matrix = movie_matrix.fillna(0)

# SVD
svd = TruncatedSVD(n_components=50)
latent_matrix = svd.fit_transform(movie_matrix_centered)

similaritySVD = latent_matrix @ latent_matrix.T

# Cosine
similarity = cosine_similarity(movie_matrix_centered)

# Normalize
scaler = MinMaxScaler()
similarity = scaler.fit_transform(similarity)
similaritySVD = scaler.fit_transform(similaritySVD)

movie_indices = {movie_id: i for i, movie_id in enumerate(movie_matrix.index)}
title_to_id = dict(zip(movies["title"], movies["movieId"]))

# Save everything

with open("models/similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)
with open("models/similaritySVD.pkl", "wb") as f:
    pickle.dump(similaritySVD, f)
with open("models/genre_similarity.pkl", "wb") as f:
    pickle.dump(genre_similarity, f)
with open("models/movie_indices.pkl", "wb") as f:
    pickle.dump(movie_indices, f)
with open("models/genre_indices.pkl", "wb") as f:
    pickle.dump(genre_indices, f)
with open("models/title_to_id.pkl", "wb") as f:
    pickle.dump(title_to_id, f)
with open("models/movie_matrix_index.pkl", "wb") as f:
    pickle.dump(movie_matrix.index, f)
with open("models/tag_similarity.pkl","wb") as f:
    pickle.dump(tag_similarity,f)
with open("models/tag_indices.pkl","wb") as f:
    pickle.dump(tag_indices,f)

print("Model saved successfully")
