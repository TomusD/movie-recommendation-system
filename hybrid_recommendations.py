from collections import defaultdict
from user_recommendations import user_user_recommendations
from item_recommendations import item_item_recommendations
from tag_recommendations import tag_recommendations
from config import movie_names, number_of_recommendations

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