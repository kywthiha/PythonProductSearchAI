from elasticsearch import Elasticsearch

from image_extract_feature import extract_features, base_model

client = Elasticsearch(hosts='http://localhost:9200',
                       basic_auth=("elastic", 'password'))

user_image_path = 'user_image.jpg'

user_image_feature = extract_features(user_image_path, base_model)
index_name = 'image_vector_34'

# Perform a cosine similarity search
search_results = client.search(index=index_name,
                               knn={"field": "image_vector", "query_vector": user_image_feature, "k": 3,
                                    "num_candidates": 100})

# Process and print search results
for hit in search_results['hits']['hits']:
    print(f"Document ID: {hit['_source']['title']}, Score: {hit['_score']}")
