from flask import Flask, render_template, request
from user_recommendations import user_user_recommendations
from item_recommendations import item_item_recommendations
from tag_recommendations import tag_recommendations
from hybrid_recommendations import hybrid_recommendations
from user_recommendations import user_similarity_jaccard, user_similarity_dice, user_similarity_cosine, user_similarity_pearson
from item_recommendations import item_similarity_jaccard, item_similarity_dice, item_similarity_cosine, item_similarity_pearson
from tag_recommendations import jaccard_similarity_tag, dice_similarity_tag, cosine_similarity_tag, pearson_similarity_tag

app = Flask(__name__)

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