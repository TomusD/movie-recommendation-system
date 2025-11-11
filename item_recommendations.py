import math
from collections import defaultdict
from config import user_movie_ratings, movie_names

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
        similarity_scores[movie1] = dict(sorted(similarity_scores[movie1].items(), key=lambda item: item[1], reverse=True)[:128]) #top_k
    
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

    top_recommendations = dict(sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)[:100])  #number of recommendations

    # Map movie IDs to names
    top_recommendations_with_titles = {movie_names[movie]: score for movie, score in top_recommendations.items() if movie in movie_names}

    return top_recommendations_with_titles

# Calculate item similarity using diffrent similarity metrices
item_similarity_jaccard = calculate_item_similarity(item_similarity_measure = 'jaccard')
item_similarity_dice = calculate_item_similarity(item_similarity_measure = 'dice')
item_similarity_cosine = calculate_item_similarity(item_similarity_measure = 'cosine')
item_similarity_pearson = calculate_item_similarity(item_similarity_measure = 'pearson')