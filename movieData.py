import os


import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
apiKey = os.getenv("API_KEY")
url = (
    f"https://api.themoviedb.org/3/movie/popular?api_key={apiKey}&language=en-US&page=1"
)


def fetchMovieData():
    response = requests.get(url)
    data = response.json()
    processedData = []
    for movie in data["results"]:
        processedData.append(
            {"title": movie["title"], "description": movie["overview"]}
        )

    movieDataFrame = pd.DataFrame(processedData)
    return movieDataFrame


if __name__ == "__main__":
    movieData = fetchMovieData()
    print(movieData.head())
