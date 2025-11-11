import csv
from collections import defaultdict

ratings_csv_path = 'ml-latest-small/ratings.csv'
movies_csv_path = 'ml-latest/movies.csv'
tags_csv_path = 'ml-latest/tags.csv'

# Defaultdict initialization
user_movie_ratings = defaultdict(lambda: defaultdict(float))
tag_movie_ratings = defaultdict(dict)
movie_names = defaultdict(str)

# Load datasets from CSV into defaultdicts
def loadRatingsData(ratings_csv_path):
    with open(ratings_csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            user, movie, rating, timestamp = row
            user_movie_ratings[user][movie] = float(rating)
    return user_movie_ratings

def loadMoviesData(movies_csv_path):
    with open(movies_csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            movieId,title,genres = row
            movie_names[movieId] = title
    return movie_names

def loadTagData(tags_csv_path):
    with open(tags_csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            userId, movieId, tag, timestamp = row 
            tag_movie_ratings[movieId][tag] = tag_movie_ratings[movieId].get(tag, 0) + 1
    return tag_movie_ratings

data_ratings = loadRatingsData(ratings_csv_path)
movie_names = loadMoviesData(movies_csv_path)
movie_tags = loadTagData(tags_csv_path)