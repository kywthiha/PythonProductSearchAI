import glob
import os

from elasticsearch import Elasticsearch

from image_extract_feature import extract_features, base_model

client = Elasticsearch(hosts='http://localhost:9200',
                       basic_auth=("elastic", 'password'))

product_images_folder_path = 'C:\\Users\\kyawt\\Documents\\GitHub\\fileserver\\uploads\\productMainImage'
product_images = glob.glob(os.path.join(product_images_folder_path, '*.jpg'))

index_name = 'image_vector_34'
# client.indices.create(index=index_name, mappings={"properties": {
#     "image_vector": {
#         "type": 'dense_vector',
#         "dims": 512,
#         "index": True,
#         "similarity": 'l2_norm'
#     },
#
#     "title": {
#         "type": 'text',
#         "fields": {
#             "keyword": {
#                 "type": "keyword"
#             }
#         }
#     },
# }})

for image_path in product_images:
    image_feature = extract_features(image_path, base_model)
    print(os.path.basename(image_path))
    resp = client.index(index=index_name, document={
        "image_vector": image_feature,
        "title": os.path.basename(image_path),
    })
    print(resp)
