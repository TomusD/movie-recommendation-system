import math
from collections import defaultdict
from config import user_movie_ratings, movie_names

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
        similarity_scores[user1] = dict(sorted(similarity_scores[user1].items(), key=lambda item: item[1], reverse=True)[:128]) #top_k

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
    
 
    top_recommendations = dict(sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)[:100])  #number of recommendations
    
    # Map movie IDs to names
    top_recommendations_with_titles = {movie_names[movie]: score for movie, score in top_recommendations.items()}
    
    return top_recommendations_with_titles

# Calculate user similarity using diffrent similarity metrices
user_similarity_jaccard = calculate_user_similarity(user_similarity_measure ='jaccard')
user_similarity_dice = calculate_user_similarity(user_similarity_measure ='dice')
user_similarity_cosine = calculate_user_similarity(user_similarity_measure ='cosine')
user_similarity_pearson = calculate_user_similarity(user_similarity_measure ='pearson')
