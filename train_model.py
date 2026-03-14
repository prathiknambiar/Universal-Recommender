import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD
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
genre_indices = {movie_id: i for i, movie_id in enumerate(movies["movieId"])}
user_means = movie_matrix.mean(axis=0)
movie_matrix_centered = movie_matrix.sub(user_means, axis=1).fillna(0)

tfidf_tag = TfidfVectorizer(stop_words='english')
tag_matrix = tfidf_tag.fit_transform(movies_with_tags["tag"])
tag_indices = {movie_id: i for i, movie_id in enumerate(movies_with_tags["movieId"])}

movie_matrix = movie_matrix.fillna(0)

svd = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = svd.fit_transform(movie_matrix_centered)

latent_matrix = latent_matrix.astype("float32")



movie_indices = {movie_id: i for i, movie_id in enumerate(movie_matrix.index)}
title_to_id = dict(zip(movies["title"], movies["movieId"]))

with open("models/latent_matrix.pkl", "wb") as f:
    pickle.dump(latent_matrix, f)
with open("models/movie_indices.pkl", "wb") as f:
    pickle.dump(movie_indices, f)
with open("models/genre_indices.pkl", "wb") as f:
    pickle.dump(genre_indices, f)
with open("models/title_to_id.pkl", "wb") as f:
    pickle.dump(title_to_id, f)
with open("models/movie_matrix_index.pkl", "wb") as f:
    pickle.dump(movie_matrix.index, f)
with open("models/tag_indices.pkl","wb") as f:
    pickle.dump(tag_indices,f)
with open("models/genre_matrix.pkl","wb") as f:
    pickle.dump(genre_matrix,f)
with open("models/tag_matrix.pkl","wb") as f:
    pickle.dump(tag_matrix,f)

print("Model saved successfully")
