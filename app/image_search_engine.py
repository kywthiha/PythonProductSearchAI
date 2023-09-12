from elasticsearch import Elasticsearch

from feature_extraction import FeatureExtractor


class ImageSearchEngine:
    def __init__(self, index_name, feature_extractor: FeatureExtractor, elasticsearch_host='http://localhost:9200',
                 elasticsearch_auth=("elastic", 'password')):
        self.client = Elasticsearch(hosts=elasticsearch_host, basic_auth=elasticsearch_auth)
        self.index_name = index_name
        self.feature_extractor = feature_extractor

    def perform_similarity_search(self, user_image_path, k=10, num_candidates=10000):
        user_image_feature = self.feature_extractor.extract_features(user_image_path)

        # Perform a cosine similarity search
        search_results = self.client.search(index=self.index_name,
                                            knn={"field": "image_vector", "query_vector": user_image_feature, "k": k,
                                                 "num_candidates": num_candidates})

        return search_results['hits']['hits']


if __name__ == "__main__":
    index_name = 'product_image_search'
    user_image_path = '../user_image.jpg'

    resnet_feature_extractor = FeatureExtractor('ResNet50')

    search_engine = ImageSearchEngine(index_name, resnet_feature_extractor)

    # Perform similarity search
    search_results = search_engine.perform_similarity_search(user_image_path)

    # Process and print search results
    for hit in search_results:
        title = hit['_source']['title']
        score = hit['_score']
        print(f"Document ID: {title}, Score: {score}")
