import math
from config import tag_movie_ratings, movie_names

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

    recommendations = similarities[:100]

    top_recommendations_with_titles = {movie_names[movie]: score for movie, score in recommendations}
    

    return top_recommendations_with_titles