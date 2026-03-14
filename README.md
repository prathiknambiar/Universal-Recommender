![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

# Smart Recommender System

A hybrid recommendation system for movies and music built with Streamlit.  
The system combines collaborative filtering, latent embeddings, and content-based features to generate personalized recommendations.

Live Application:  
https://universal-recommender.streamlit.app/

---

## Overview

This project implements a recommendation system capable of suggesting:

- Movies based on user ratings and metadata
- Songs based on audio features and similarity

The system integrates multiple recommendation strategies to improve accuracy and relevance.

---

## Features

- Movie recommendation using latent embeddings and similarity scoring
- Music recommendation based on audio feature similarity
- Hybrid recommendation combining collaborative and content-based signals
- Poster and album artwork retrieval via external APIs
- Web interface built with Streamlit

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Requests

---

## Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd <PROJECT_FOLDER>
```
Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the streamlit application:
```bash
streamlit run app.py
```

---

## Model Training Section (for `train_model.py`)

```markdown
Model Training

The recommendation models were trained using the MovieLens dataset.

To retrain the models locally:

```bash
python train_model.py
```
```

