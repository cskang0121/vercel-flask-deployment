# RDS
# admin
# xMOUY8UN96jeyquMjhnh
# cs425-nagg-recommender-db.cospw8wfrk1d.us-east-2.rds.amazonaws.com

import os
import socket
import json
import math

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

from Engine import Engine


# Flask application setup
app = Flask(__name__)

# MySQL database setup
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:1234@localhost:3306/recommender'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://admin:xMOUY8UN96jeyquMjhnh@cs425-nagg-recommender-db.cospw8wfrk1d.us-east-2.rds.amazonaws.com:3306/recommender'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_size': 100, 'pool_recycle': 280}
db = SQLAlchemy(app)
CORS(app)

# ORM setup
class UtilityMatrix(db.Model):
    __tablename__ = 'utility_matrix'
    user_id = db.Column(db.String(), primary_key=True, nullable=False)
    user_preferences = db.Column(db.String(), nullable=False)
    user_profile = db.Column(db.String(), nullable=False)
    news_count = db.Column(db.Integer, nullable=False)

    def __init__(self, user_id, user_preferences, user_profile, news_count):
        self.user_id = user_id
        self.user_preferences = user_preferences
        self.user_profile = user_profile
        self.news_count = news_count

    def to_dict(self):
        return {
            'user_id': self.user_id,
            'user_preferences': self.user_preferences,
            'user_profile': self.user_profile,
            'news_count': self.news_count
        }


# Recommender engine setup
recommender_engine = Engine()

# Generate news embeddings for ALL news ##
# recommender_engine.generate_embedding()#
##########################################

# Retrieve recommendations
@app.route('/recommendations')
def retrieve_k_recommendations():

    user_id = request.args['user_id']

    # Total no. of content-based recommendations to return
    k = int(request.args['k'])
    
    record = db.session.scalars(
        db.select(UtilityMatrix).filter_by(user_id=user_id).limit(1)
    ).first()

    # Return random recommendations if new user -> For news exploration
    if not record:
        
        return {
            'recommendation_news': recommender_engine.get_k_news(k),
        }, 200
    
    # Convert JSON string to user preferences
    user_pref = json.loads(record.user_preferences)

    # Convert JSON string to user profile
    user_profile = json.loads(record.user_profile)
    news_count = record.news_count

    # Perform normalisation on user profile
    user_profile = [x / news_count for x in user_profile] if news_count else user_profile

    # Return if no preferences
    if len(set(list(user_pref.values()))) == 1:
        return {
            'recommendation_news': recommender_engine.get_k_news(k),
        }, 200

    # Call recommender engine for customised recommendation
    recommendations = recommender_engine.top_k_recommendation(user_pref, user_profile, k)

    return ({'recommendations': recommendations}), 200


# Update user preferences
@app.route('/user-preferences', methods=['PUT'])
def update_user_preferences():

    user_id = request.args['user_id']
    news_id = request.args['news_id']
    category = request.args['category']

    record = db.session.scalars(
        db.select(UtilityMatrix).filter_by(user_id=user_id).limit(1)
    ).first()

    # Create record in utility matrix if new user
    if not record:
        new_user_preferences = '{"Latest News": 0, "Asia": 0, "Business": 0, "Singapore News": 0, "Sports": 0, "World News": 0}'
        new_user_news_count = 1
        new_user_profile = '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'

        new_user = UtilityMatrix(user_id, new_user_preferences, new_user_profile, new_user_news_count)
        db.session.add(new_user)
        db.session.commit()

        record = db.session.scalars(
            db.select(UtilityMatrix).filter_by(user_id=user_id).limit(1)
        ).first()

    # Retrieve news embedding of provided news id from Firebase
    news_emb = recommender_engine.get_embedding_by_id(news_id)

    user_pref = json.loads(record.user_preferences)
    user_profile = json.loads(record.user_profile)
    news_count = record.news_count

    # Update user preferences - give heavier weight on the interested topic
    user_pref[category] = user_pref[category] + math.log(1 + user_pref[category]) if user_pref[category] else 1
    record.user_preferences = json.dumps(user_pref)

    # Update user profile
    user_profile = [x + y for x, y in zip(news_emb, user_profile)]
    record.user_profile = json.dumps(user_profile)

    # Update news count
    news_count += 1
    record.news_count = news_count

    db.session.commit()

    # Return updated user information
    return {
        'updated_user_pref'     : user_pref,
        'updated_user_profile'  : user_profile,
        'updated_news_count'    : news_count
    }, 200    


@app.route('/health')
def health():
    return 'Healthy...', 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
    