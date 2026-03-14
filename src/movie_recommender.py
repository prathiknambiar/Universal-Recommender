import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("data/movielens/movie.csv")

with open("models/latent_matrix.pkl", "rb") as f:
    latent_matrix = pickle.load(f)
with open("models/movie_indices.pkl", "rb") as f:
    movie_indices = pickle.load(f)
with open("models/title_to_id.pkl", "rb") as f:
    title_to_id = pickle.load(f)
with open("models/movie_matrix_index.pkl", "rb") as f:
    movie_index = pickle.load(f)
with open("models/genre_indices.pkl", "rb") as f:
    genre_indices = pickle.load(f)
with open("models/tag_indices.pkl","rb") as f:
    tag_indices = pickle.load(f)
with open("models/genre_matrix.pkl", "rb") as f:
    genre_matrix = pickle.load(f)
with open("models/tag_matrix.pkl", "rb") as f:
    tag_matrix = pickle.load(f)

id_to_title = dict(zip(movies["movieId"], movies["title"]))

def recommend(movie_title):

    movie_id = title_to_id.get(movie_title)
    if movie_id is None:
        return []

    idx = movie_indices[movie_id]
    genre_idx = genre_indices[movie_id]
    tag_idx = tag_indices[movie_id]
    scores = cosine_similarity(
        latent_matrix[idx].reshape(1, -1),
        latent_matrix
        ).flatten()
    genre_scores = cosine_similarity(
        genre_matrix[genre_idx],
        genre_matrix
    ).flatten()

    tag_scores = cosine_similarity(
        tag_matrix[tag_idx],
        tag_matrix
    ).flatten()
    top_movies = list(enumerate(scores))
    top_movies = sorted(top_movies, key=lambda x: x[1], reverse=True)[1:80]

    reranked = []
    for i, _ in top_movies:
        final_score = (
            0.5 * scores[i] +
            0.25 * tag_scores[i]+
            0.25 * genre_scores[i] 
        )

        reranked.append((i, final_score))

    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)

    movie_ids = [movie_index[i[0]] for i in reranked[:5]]

    return [id_to_title[mid] for mid in movie_ids]
