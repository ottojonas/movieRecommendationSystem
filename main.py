import pandas as pd
from icecream import ic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


from movieData import fetchMovieData


def makeRecommendations(title, tfidf_matrix, indices, cosine_sim):
    if title not in indices:
        print(f"Title '{title}' not found in movie dataset.")
        return []
    idx = indices[title]
    simScores = list(enumerate(cosine_sim[idx]))
    simScores = sorted(simScores, key=lambda x: x[1], reverse=True)
    simScores = simScores[1:11]
    movieIndices = [i[0] for i in simScores]
    print(f"Recommended movies for '{title}': ")
    return movieDataFrame["title"].iloc[movieIndices]


movieDataFrame = fetchMovieData()

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movieDataFrame["description"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movieDataFrame.index, index=movieDataFrame["title"]).to_dict()

userMovieInput = input("Enter the movie title for recommendations: ").strip()

if userMovieInput:
    recommendations = makeRecommendations(
        userMovieInput, tfidf_matrix, indices, cosine_sim
    )
    if (hasattr(recommendations, "empty") and recommendations.empty) or (
        not hasattr(recommendations, "empty") and not recommendations
    ):
        print("No recommendations found, please enter another movie")
    else:
        ic(recommendations)

else:
    print("No movie title entered")
