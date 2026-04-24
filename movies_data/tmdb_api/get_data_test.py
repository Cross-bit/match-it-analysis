#!/bin/python3
import requests
import requests

# ====================
# DESCRIPTION
# ====================
# Test script to obtain basic info from TMDB.
#

API_TOKEN=""

# Search for movie
def search_movie(query):
    url = 'https://api.themoviedb.org/3/search/movie'
    params = {
        'api_key': API_TOKEN,
        'query': query,
        'language': 'en-US'
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        for movie in results:
            print(f"{movie['title']} ({movie['release_date']}) - ID: {movie['id']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Gets movie details
def get_movie_details(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}'
    params = {
        'api_key': API_TOKEN,
        'language': 'en-US'
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        movie = response.json()
        print(movie)
    else:
        print(f"Error: {response.status_code} - {response.text}")


def get_movie_credits(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/credits'
    params = {
        'api_key': API_TOKEN,
        'language': 'en-US'
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        credits = response.json()

        # Get director
        directors = [member['name'] for member in credits['crew'] if member['job'] == 'Director']
        cast = [actor['name'] for actor in credits['cast'][:5]]  # Top 5 actors

        print(f"\nDirector(s): {', '.join(directors) if directors else 'Unknown'}")
        print(f"Top Cast: {', '.join(cast) if cast else 'Unknown'}")
    else:
        print(f"Error fetching credits: {response.status_code} - {response.text}")


if __name__ == '__main__':
    movie_id = 10478
    get_movie_details(movie_id)
    get_movie_credits(movie_id)

