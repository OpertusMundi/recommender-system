import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


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


class Recommender_contents():
    def __init__(self):
        self.datasets = None
        self.load_data()

    def load_data(self):
        # Reading datsets file:
        self.datasets = pd.read_csv('data/datasets.csv', sep=',', encoding='utf-8')

    def similarity_contents(self, dataset_id=144, model='RotatE', number_of_recommendations=3):
        if model == 'RotatE':
            path = "content_based_recommendations/EmbeddingModels/results_official/resultsRotatE/"
        elif model == 'TransH':
            path = "content_based_recommendations/EmbeddingModels/results_official/resultsTransH/"

        all_datasets_ids = self.datasets.dataset_id
        path = path + "trained_model.pkl"
        model = torch.load(path)
        entity_embeddings = model.entity_representations[0]
        original = entity_embeddings(torch.as_tensor(dataset_id)).detach().numpy()
        d = dict.fromkeys(all_datasets_ids)
        for i in range(len(all_datasets_ids)):
            embdding = entity_embeddings(torch.as_tensor(all_datasets_ids[i])).detach().numpy()
            # print("Is embedding complex(real and imaginary) in nature?", np.iscomplexobj(embdding))  # -> False
            cos_sim = cosine_similarity(original.reshape(1, -1), embdding.reshape(1, -1))
            d[all_datasets_ids[i]] = cos_sim

        recommended_ids = sorted(d, key=d.get, reverse=True)[1:number_of_recommendations + 1]
        return recommended_ids

    def recommend_datasets_on_contents(self, dataset_id, model ,number_of_recommendations=1):
        n = number_of_recommendations * -1
        return self.similarity_contents(dataset_id, model, number_of_recommendations=number_of_recommendations)


if __name__ == "__main__":
    recommender = Recommender()
    recommender_contents = Recommender_contents()
    result = recommender.recommend_by_user_id(user_id=5, number_of_recommendations=4)
    print("result ", result)
    result2 = recommender_contents.recommend_datasets_on_contents(dataset_id=144, number_of_recommendations=4)
    print("result2 ", result2)
