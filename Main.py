from flask import Flask, render_template, request
from collections import defaultdict
import csv
import math

app = Flask(__name__)

ratings_csv_path = 'ml-latest-small/ratings.csv'
movies_csv_path = 'ml-latest/movies.csv'
tags_csv_path = 'ml-latest/tags.csv'
top_k = 128
number_of_recommendations = 100

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


#Calculate similarity scores between two users based on their common rated movies
def jaccard_similarity_user(user1_ratings, user2_ratings):
    set_user1 = set(user1_ratings.keys())
    set_user2 = set(user2_ratings.keys())
    intersection = len(set_user1.intersection(set_user2))
    union = len(set_user1.union(set_user2))
    jaccard_sim = intersection / union if union != 0 else 0
    return jaccard_sim

def dice_similarity_user(user1_ratings, user2_ratings):
    set_user1 = set(user1_ratings.keys())
    set_user2 = set(user2_ratings.keys())
    intersection = len(set_user1.intersection(set_user2))
    dice_sim = 2 * intersection / (len(set_user1) + len(set_user2))
    return dice_sim

def cosine_similarity_user(user1_ratings, user2_ratings):
    common_movies = set(user1_ratings.keys()).intersection(user2_ratings.keys())

    if not common_movies:
        return 0

    dot_product = sum(user1_ratings[movie] * user2_ratings[movie] for movie in common_movies)
    norm_ratings1 = math.sqrt(sum(user1_ratings[movie] ** 2 for movie in common_movies))
    norm_ratings2 = math.sqrt(sum(user2_ratings[movie] ** 2 for movie in common_movies))

    cosine_sim = dot_product / (norm_ratings1 * norm_ratings2) if norm_ratings1 * norm_ratings2 != 0 else 0
    return cosine_sim

def pearson_similarity_user(user1_ratings, user2_ratings):
    common_movies = set(user1_ratings.keys()).intersection(user2_ratings.keys())

    if not common_movies:
        return 0

    mean_ratings1 = sum(user1_ratings[movie] for movie in common_movies) / len(common_movies)
    mean_ratings2 = sum(user2_ratings[movie] for movie in common_movies) / len(common_movies)

    numerator = sum((user1_ratings[movie] - mean_ratings1) * (user2_ratings[movie] - mean_ratings2) for movie in common_movies)
    denominator1 = math.sqrt(sum((user1_ratings[movie] - mean_ratings1)**2 for movie in common_movies))
    denominator2 = math.sqrt(sum((user2_ratings[movie] - mean_ratings2)**2 for movie in common_movies))

    pearson_sim = numerator / (denominator1 * denominator2) if denominator1 * denominator2 != 0 else 0
    return pearson_sim

# Calculate the user similarity scores
def calculate_user_similarity(user_similarity_measure):
    
    similarity_scores = defaultdict(dict)
    users = list(user_movie_ratings.keys())
    

    if user_similarity_measure == 'jaccard':
        similarity_function = jaccard_similarity_user
    elif user_similarity_measure == 'dice':
        similarity_function = dice_similarity_user
    elif user_similarity_measure == 'cosine':
        similarity_function = cosine_similarity_user
    elif user_similarity_measure == 'pearson':
        similarity_function = pearson_similarity_user

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user1,user2 = users[i],users[j]
            similarity = similarity_function(user_movie_ratings[user1], user_movie_ratings[user2])
            similarity_scores[user1][user2] = similarity

    # Complete the similarity matrix by copying the values to the other triangle
    for user1 in users:
        for user2, similarity in similarity_scores[user1].items():
            similarity_scores[user2][user1] = similarity

    # Get the top N similar users for each user
    for user1 in users:
        similarity_scores[user1] = dict(sorted(similarity_scores[user1].items(), key=lambda item: item[1], reverse=True)[:top_k])

    return similarity_scores

# Generate movie recommendations for a specific user based on similar users ratings
def user_user_recommendations(user_id, similarity_scores):

    user_ratings = user_movie_ratings[user_id]
    similar_users = list(similarity_scores[user_id].keys())

    # Create a set of movies the user has already rated
    rated_movies = set(user_ratings.keys())

    # Create a defaultdict to store recommendation scores
    recommendation_scores = defaultdict(float)

    # Calculate the product of similarity scores and user ratings for each movie
    for similar_user in similar_users:
        for movie in user_movie_ratings[similar_user]:
            if movie not in rated_movies:      
                if similarity_scores[user_id][similar_user] != 0:
                    recommendation_scores[movie] += (similarity_scores[user_id][similar_user] * user_movie_ratings[similar_user][movie]) / similarity_scores[user_id][similar_user]
                else:
                    recommendation_scores[movie] += 0

   # Normalize the recommendation scores to fit within the range of 0 to 5
    max_score = max(recommendation_scores.values(), default=0)
    min_score = min(recommendation_scores.values(), default=0)

    if max_score != min_score:
        scaling_factor = 5 / (max_score - min_score)
        for movie in recommendation_scores:
            recommendation_scores[movie] = (recommendation_scores[movie] - min_score) * scaling_factor
    
 
    top_recommendations = dict(sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)[:number_of_recommendations])
    
    # Map movie IDs to names
    top_recommendations_with_titles = {movie_names[movie]: score for movie, score in top_recommendations.items()}
    
    return top_recommendations_with_titles

# Calculate user similarity using diffrent similarity metrices
user_similarity_jaccard = calculate_user_similarity(user_similarity_measure ='jaccard')
user_similarity_dice = calculate_user_similarity(user_similarity_measure ='dice')
user_similarity_cosine = calculate_user_similarity(user_similarity_measure ='cosine')
user_similarity_pearson = calculate_user_similarity(user_similarity_measure ='pearson')


#Calculate similarity scores between two item sets
def jaccard_similarity_item(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_sim = intersection / union if union != 0 else 0
    return jaccard_sim

def dice_similarity_item(set1, set2):
    intersection = len(set1.intersection(set2))
    dice_sim = 2 * intersection / (len(set1) + len(set2))
    return dice_sim

def cosine_similarity_item(set1, set2):
    dot_product = sum(val1 * val2 for val1, val2 in zip(set1, set2))
    norm_set1 = math.sqrt(sum(val ** 2 for val in set1))
    norm_set2 = math.sqrt(sum(val ** 2 for val in set2))
    cosine_sim = dot_product / (norm_set1 * norm_set2) if norm_set1 * norm_set2 != 0 else 0
    return cosine_sim

def pearson_similarity_item(set1, set2):
    mean_set1 = sum(set1) / len(set1)
    mean_set2 = sum(set2) / len(set2)

    numerator = sum((val1 - mean_set1) * (val2 - mean_set2) for val1, val2 in zip(set1, set2))
    denominator1 = math.sqrt(sum((val - mean_set1) ** 2 for val in set1))
    denominator2 = math.sqrt(sum((val - mean_set2) ** 2 for val in set2))

    pearson_sim = numerator / (denominator1 * denominator2) if denominator1 * denominator2 != 0 else 0
    return pearson_sim

# Calculate item-item similarity using the specified similarity measure
def calculate_item_similarity(item_similarity_measure):

    similarity_scores = defaultdict(dict)

    # Take the movies from the ratings of the first 10 users
    first_10_users = list(user_movie_ratings.keys())[:10]
    movies = sorted(set(movie for user in first_10_users for movie in user_movie_ratings[user].keys()))

    if item_similarity_measure == 'jaccard':
        similarity_function = jaccard_similarity_item
    elif item_similarity_measure == 'dice':
        similarity_function = dice_similarity_item
    elif item_similarity_measure == 'cosine':
        similarity_function = cosine_similarity_item
    elif item_similarity_measure == 'pearson':
        similarity_function = pearson_similarity_item

    # Loop over pairs of movies to calculate similarity scores
    for i in range(len(movies)):
        for j in range(i + 1, len(movies)):
            movie1, movie2 = movies[i], movies[j]
            # Get the users who rated either movie
            users_rated_either = set(user for user in user_movie_ratings if movie1 in user_movie_ratings[user] or movie2 in user_movie_ratings[user])
            # Calculate similarity for users who rated either movie
            if users_rated_either:
                set1 = set(user_movie_ratings[user][movie1] if movie1 in user_movie_ratings[user] else 0 for user in users_rated_either)
                set2 = set(user_movie_ratings[user][movie2] if movie2 in user_movie_ratings[user] else 0 for user in users_rated_either)
                similarity = similarity_function(set1, set2)
                similarity_scores[movie1][movie2] = similarity
    
    # Copy values to the lower triangular part to ensure symmetry
    for i in range(len(movies)):
        for j in range(i + 1, len(movies)):
            movie1, movie2 = movies[i], movies[j]
            similarity_scores[movie2][movie1] = similarity_scores[movie1][movie2]

    for movie1 in movies:
        similarity_scores[movie1] = dict(sorted(similarity_scores[movie1].items(), key=lambda item: item[1], reverse=True)[:top_k])
    
    return similarity_scores

# Generate movie recommendations for a specific user based on item-item similarity
def item_item_recommendations(user_id, similarity_scores):

    user_ratings = user_movie_ratings[user_id]

    # Create a set of movies the user has already rated
    rated_movies = set(user_ratings.keys())

    # Create a defaultdict to store recommendation scores
    recommendation_scores = defaultdict(float)

    for movie in user_movie_ratings[user_id]:
        similar_movies = list(similarity_scores[movie].keys())
        # Calculate recommendation score for the movie
        for similar_movie in similar_movies:
            # Check if the similar movie is already rated
            if similar_movie not in rated_movies:
                if similarity_scores[movie][similar_movie] != 0:
                    recommendation_scores[similar_movie] += (similarity_scores[movie][similar_movie] * user_ratings[movie]) / similarity_scores[movie][similar_movie]
                else:
                    recommendation_scores[similar_movie] = 0

    # Normalize the recommendation scores to fit within the range of 0 to 5
    max_score = max(recommendation_scores.values(), default=0)
    min_score = min(recommendation_scores.values(), default=0)

    if max_score != min_score:
        scaling_factor = 5 / (max_score - min_score)
        for movie in recommendation_scores:
            recommendation_scores[movie] = (recommendation_scores[movie] - min_score) * scaling_factor

    top_recommendations = dict(sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)[:number_of_recommendations])

    # Map movie IDs to names
    top_recommendations_with_titles = {movie_names[movie]: score for movie, score in top_recommendations.items() if movie in movie_names}

    return top_recommendations_with_titles

# Calculate item similarity using diffrent similarity metrices
item_similarity_jaccard = calculate_item_similarity(item_similarity_measure = 'jaccard')
item_similarity_dice = calculate_item_similarity(item_similarity_measure = 'dice')
item_similarity_cosine = calculate_item_similarity(item_similarity_measure = 'cosine')
item_similarity_pearson = calculate_item_similarity(item_similarity_measure = 'pearson')


# Calculate similarities between two tag vectors
def jaccard_similarity_tag(tag_vector1, tag_vector2):
    set_tags1 = set(tag_vector1.keys())
    set_tags2 = set(tag_vector2.keys())
    intersection = len(set_tags1.intersection(set_tags2))
    union = len(set_tags1.union(set_tags2))
    jaccard_sim = intersection / union if union != 0 else 0
    return jaccard_sim

def dice_similarity_tag(tag_vector1, tag_vector2):
    set_tags1 = set(tag_vector1.keys())
    set_tags2 = set(tag_vector2.keys())
    intersection = len(set_tags1.intersection(set_tags2))
    dice_sim = 2 * intersection / (len(set_tags1) + len(set_tags2))
    return dice_sim

def cosine_similarity_tag(tag_vector1, tag_vector2):
    dot_product = sum(tag_vector1[tag] * tag_vector2[tag] for tag in set(tag_vector1) & set(tag_vector2))
    norm_tags1 = math.sqrt(sum(val ** 2 for val in tag_vector1.values()))
    norm_tags2 = math.sqrt(sum(val ** 2 for val in tag_vector2.values()))
    cosine_sim = dot_product / (norm_tags1 * norm_tags2) if norm_tags1 * norm_tags2 != 0 else 0
    return cosine_sim

def pearson_similarity_tag(tag_vector1, tag_vector2):
    mean_tags1 = sum(tag_vector1.values()) / len(tag_vector1)
    mean_tags2 = sum(tag_vector2.values()) / len(tag_vector2)

    numerator = sum((tag_vector1[tag] - mean_tags1) * (tag_vector2[tag] - mean_tags2) for tag in set(tag_vector1) & set(tag_vector2))
    denominator1 = math.sqrt(sum((tag_vector1[tag] - mean_tags1) ** 2 for tag in tag_vector1))
    denominator2 = math.sqrt(sum((tag_vector2[tag] - mean_tags2) ** 2 for tag in tag_vector2))

    pearson_sim = numerator / (denominator1 * denominator2) if denominator1 * denominator2 != 0 else 0
    return pearson_sim

# Get the tag vector for a given movie
def get_tag_vector(movie_id):
    return tag_movie_ratings.get(movie_id, {})

# Calculate the most similar movies and return the number of recommendations
def tag_recommendations(movie_id, similarity_func):

    target_tag_vector = get_tag_vector(movie_id)
    similarities = []

    for candidate_movie_id, candidate_tag_vector in tag_movie_ratings.items():
        if candidate_movie_id != movie_id:
            similarity = similarity_func(target_tag_vector, candidate_tag_vector)
            similarities.append((candidate_movie_id, similarity))

    # Sort movies based on similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Normalize the recommendation scores within the range of 0 to 5
    max_score = similarities[0][1] if similarities else 0
    min_score = similarities[-1][1] if similarities else 0

    if max_score != min_score:
        scaling_factor = 5 / (max_score - min_score)
        for i in range(len(similarities)):
            movie, score = similarities[i]
            similarities[i] = (movie, (score - min_score) * scaling_factor)

    recommendations = similarities[:number_of_recommendations]

    top_recommendations_with_titles = {movie_names[movie]: score for movie, score in recommendations}
    

    return top_recommendations_with_titles


# Calculate the most similar movies based on the 3 algorithms and return the number of recommendations
def hybrid_recommendations(user_id, user_similarity, item_similarity, tag_similarity, w1 = 0.4, w2 = 0.3, w3 = 0.7):

    # Get recommendations from user-user recommendation algorithm
    user_user_rec = user_user_recommendations(user_id, user_similarity)

    # Map movie names to IDs
    movie_name_to_movie_id = {movie_id: score for movie_id, movie_name in movie_names.items()
                        for movie_name_key, score in user_user_rec.items() if movie_name == movie_name_key}
    movieid_top_recommendations = dict(sorted(movie_name_to_movie_id.items(), key=lambda item: item[1], reverse=True))

    # Get recommendations from item-item recommendation algorithm
    item_item_rec = item_item_recommendations(user_id, item_similarity)
    
    # Get recommendations from tag-based recommendation algorithm
    tag_rec = tag_recommendations((next(iter(movieid_top_recommendations))), tag_similarity)

    # Combine recommendations
    hybrid_rec = defaultdict(float)

    for movie, score in user_user_rec.items():
        hybrid_rec[movie] += w1 * score

    for movie, score in item_item_rec.items():
        hybrid_rec[movie] += w2 * score

    for movie, score in tag_rec.items():
        hybrid_rec[movie] += w3 * score

    # Normalize the recommendation scores to fit within the range of 0 to 5
    max_score = max(hybrid_rec.values(), default=0)
    min_score = min(hybrid_rec.values(), default=0)

    if max_score != min_score:
        scaling_factor = 5 / (max_score - min_score)
        for movie in hybrid_rec:
            hybrid_rec[movie] = (hybrid_rec[movie] - min_score) * scaling_factor

    top_hybrid_rec = dict(sorted(hybrid_rec.items(), key=lambda item: item[1], reverse=True)[:number_of_recommendations])
    
    return top_hybrid_rec


@app.route('/')
def index():
    return render_template('index.html')

# Route for handling form submission and generating recommendations
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    input_id = request.form['input_id']
    similarity_measure = request.form['similarity_measure']
    algorithm = request.form['algorithm']
    recommendations = {}

    # User-User algorithm
    if algorithm == 'user_user':
        if similarity_measure == 'jaccard':
            user_similarity = user_similarity_jaccard
        elif similarity_measure == 'dice':
            user_similarity = user_similarity_dice
        elif similarity_measure == 'cosine':
            user_similarity = user_similarity_cosine
        elif similarity_measure == 'pearson':
            user_similarity = user_similarity_pearson

        recommendations = user_user_recommendations(input_id, user_similarity)
    # Item-Item algorithm
    elif algorithm == 'item_item':
        if similarity_measure == 'jaccard':
            item_similarity = item_similarity_jaccard
        elif similarity_measure == 'dice':
            item_similarity = item_similarity_dice
        elif similarity_measure == 'cosine':
            item_similarity = item_similarity_cosine
        elif similarity_measure == 'pearson':
            item_similarity = item_similarity_pearson

        recommendations = item_item_recommendations(input_id, item_similarity)

    # Tag based algorithm
    elif algorithm == 'tag':
        if similarity_measure == 'jaccard':
            tag_similarity = jaccard_similarity_tag
        elif similarity_measure == 'dice':
            tag_similarity = dice_similarity_tag
        elif similarity_measure == 'cosine':
            tag_similarity = cosine_similarity_tag
        elif similarity_measure == 'pearson':
            tag_similarity = pearson_similarity_tag

        recommendations = tag_recommendations(input_id, tag_similarity)
    
    # Hybrid algorithm
    elif algorithm == 'hybrid':
        if similarity_measure == 'jaccard':
            user_similarity = user_similarity_jaccard
            item_similarity = item_similarity_jaccard
            tag_similarity = jaccard_similarity_tag
        elif similarity_measure == 'dice':
            user_similarity = user_similarity_dice
            item_similarity = item_similarity_dice
            tag_similarity = dice_similarity_tag
        elif similarity_measure == 'cosine':
            user_similarity = user_similarity_cosine
            item_similarity = item_similarity_cosine
            tag_similarity = cosine_similarity_tag
        elif similarity_measure == 'pearson':
            user_similarity = user_similarity_pearson
            item_similarity = item_similarity_pearson
            tag_similarity = pearson_similarity_tag

        recommendations = hybrid_recommendations(input_id, user_similarity, item_similarity, tag_similarity, w1 = 0.4, w2 = 0.3, w3 = 0.7)

    # Render the recommendations template
    return render_template('recommendations.html', input_id=input_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)