import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from content_based_recommendations.recommendations import similarity



class Recommender:
    """
    Helper class for calculating predictions for recommending assets
    """

    def __init__(self):
        self.assets = None
        self.users = None
        self.ratings = None
        self.user_prediction = None
        self.asset_prediction = None
        self.popular_assets = None
        self.load_data()
        self.generate_predictions()
        self.generate_popular_assets()

    # TODO: Replace data loading mechanism with API calls
    def load_data(self):
        # Reading users file:
        self.users = pd.read_csv('data/users.csv', sep=',', encoding='utf-8')
        # Reading ratings file:
        self.ratings = pd.read_csv('data/ratings.csv', sep=',', encoding='utf-8')
        # Reading items file:
        self.assets = pd.read_csv('data/assets.csv', sep=',', encoding='utf-8')

    def create_user_asset_matrix(self):
        # Total number of unique users
        n_users = self.ratings.user_id.unique().shape[0]
        # Total number of unique assets
        n_assets = self.ratings.asset_id.unique().shape[0]
        # Creating a matrix of which user rates which asset by how much
        data_matrix = np.zeros((n_users, n_assets))
        for line in self.ratings.itertuples():
            data_matrix[line[1] - 1, line[2] - 1] = line[3]
        return data_matrix

    def similarity(self, user_asset_matrix, similarity_type='user'):
        if similarity_type == 'user':
            return pairwise_distances(user_asset_matrix, metric='cosine')
        elif similarity_type == 'asset':
            return pairwise_distances(user_asset_matrix.T, metric='cosine')

    def predict(self, ratings, similarity, type='user'):
        pred = 0
        if type == 'user':
            mean_user_rating = ratings.mean(axis=1).reshape(-1, 1)
            ratings_diff = (ratings - mean_user_rating)
            pred = mean_user_rating + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif type == 'asset':
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        return pred

    def generate_predictions(self):
        user_asset_matrix = self.create_user_asset_matrix()
        user_similarity = self.similarity(user_asset_matrix, similarity_type='user')
        asset_similarity = self.similarity(user_asset_matrix, similarity_type='asset')
        self.user_prediction = self.predict(user_asset_matrix, user_similarity, type='user')
        self.asset_prediction = self.predict(user_asset_matrix, asset_similarity, type='asset')

    def recommend_by_user_id(self, user_id, number_of_recommendations=1):
        n = int(number_of_recommendations * -1)
        return self.user_prediction[user_id].argsort()[n:][::-1].tolist()

    def recommend_by_asset_id(self, asset_id, number_of_recommendations=1):
        n = number_of_recommendations * -1
        return self.asset_prediction[asset_id].argsort()[n:][::-1].tolist()

    def generate_popular_assets(self):
        popular_asset_dict = {}
        for line in self.ratings.itertuples():
            if line[2] in popular_asset_dict:
                popular_asset_dict[line[2]] += line[3]
            else:
                popular_asset_dict[line[2]] = line[3]
        self.popular_assets = [asset_id for asset_id, rating in
                               sorted(popular_asset_dict.items(), key=lambda item: item[1])]

    def recommend_popular_assets(self, number_of_recommendations=1):
        return self.popular_assets[:number_of_recommendations]

    def recommend_datasets_on_contents(self, number_of_recommendations=1):
        return similarity(number_of_recommendations=number_of_recommendations)

