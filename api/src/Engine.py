import json
import math
import gensim
import random
import numpy as np

import firebase_admin
from firebase_admin import db
from firebase_admin import credentials

from ast import literal_eval

class Engine:
    '''Doc2Vec-Based Recommender Engine'''


    def __init__(self):
        self.d2v_model = gensim.models.doc2vec.Doc2Vec.load('d2v_v2.model')
        self.firebase_credentials = credentials.Certificate("firebase/cs425-news-database-firebase-adminsdk-rter4-fbdf75f1ef.json")

        firebase_admin.initialize_app(self.firebase_credentials, {
            'databaseURL': "https://cs425-news-database-default-rtdb.asia-southeast1.firebasedatabase.app/"
        })

        self.firebase_database = db.reference('/cna-api/')
        self.firebase_database_emb = db.reference('/cna-emb-api/')


    def get_all_embeddings(self):
        return self.firebase_database_emb.get()
    
    def get_embedding_by_id(self, news_id):
        res = list(self.firebase_database_emb.get().values())
        for info in res:
            if info['news_id'] == news_id:
                return json.loads(info['embedding'])
        return []

    def get_k_news(self, k):
        res = self.firebase_database.get().keys()
        return random.sample(res, k)

    def vectorise(self, processed_news) :
        return list(self.d2v_model.infer_vector(processed_news))


    def generate_embedding(self):
        self.firebase_database_emb.delete()
        
        news = list(self.firebase_database.get().items())
        
        for i in range(len(news)):
            upload_dict = {
                'news_id'   : news[i][0],
                'embedding' : str(self.vectorise(list(news[i][1]['preprocessedText'].split(' ')))),
                'category'  : news[i][1]['category']
                }

            upload_json = json.loads(json.dumps(upload_dict))

            self.firebase_database_emb.push(upload_json)
        
        return


    def top_k_recommendation(self, user_pref, user_profile, k):

        # Number of sections
        n = len(user_pref)

        # Sort the sections based on user preferences
        sorted_user_pref = sorted(user_pref.items(), key=lambda x:x[1])[::-1]
        # print(sorted_user_pref)

        # Retrive the sorted scores 
        sorted_scores = [sorted_user_pref[i][1] for i in range(n)]
        # print(sorted_scores)

        # Compute the sorted weights using softmax
        sorted_weights = self.softmax(np.array([sorted_scores]))[0]
        # print(sorted_weights)
        
        # Computes the number of recommendations (based on weights) for each sections
        section_news = {}
        cnt = 0
        for i in range(n):

            if (cnt + math.ceil(k * sorted_weights[i])) >= k:
                section_news[sorted_user_pref[i][0]] = k - cnt
                break
                
            section_news[sorted_user_pref[i][0]] = math.ceil(k * sorted_weights[i])
            
            cnt += math.ceil(k * sorted_weights[i])

        recommendations = []

        # Compoute the content based recommendations based on user profile and user preferences
        for section in section_news:
            cos_sim_rank = {}
            news = list(self.firebase_database_emb.order_by_child('category').equal_to(section).get().items())
            
            for i in range(len(news)):
                cos_sim_rank[news[i][1]['news_id']] = self.cosine_sim(user_profile, json.loads(news[i][1]['embedding']))

            recommendations += [x[0] for x in sorted(cos_sim_rank.items(), key=lambda x:x[1])][:section_news[section]]
            
        # Return recommendations and other relevant information
        return {
            'preferences'          : user_pref,
            'recommendation_ratio' : section_news,
            'recommendation_news'  : recommendations,
        }


    def cosine_sim(self, pref_vec, news_vec):

        np_pref_vector = np.array(pref_vec)
        np_news_vector = np.array(news_vec)
        
        csim = np.dot(np_pref_vector, np_news_vector) / (np.linalg.norm(np_pref_vector) * np.linalg.norm(np_news_vector))
         
        return csim


    def softmax(self, z):
        assert len(z.shape) == 2

        s = np.max(z, axis=1)
        s = s[:, np.newaxis]        

        e_x = np.exp(z - s)

        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]   
        
        return e_x / div