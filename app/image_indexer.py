import glob
import os

from elasticsearch import Elasticsearch

from feature_extraction import FeatureExtractor


class ImageIndexer:
    def __init__(self, index_name, image_folder_path, feature_extractor: FeatureExtractor,
                 elasticsearch_host='http://localhost:9200',
                 elasticsearch_auth=("elastic", 'password')):
        self.client = Elasticsearch(hosts=elasticsearch_host, basic_auth=elasticsearch_auth)
        self.index_name = index_name
        self.image_folder_path = image_folder_path
        self.feature_extractor = feature_extractor

    def create_index(self):
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, mappings={"properties": {
                "image_vector": {
                    "type": 'dense_vector',
                    "dims": 2048,
                    "index": True,
                    "similarity": 'l2_norm'
                },
                "title": {
                    "type": 'text',
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                "image_url": {
                    "type": 'text',
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
            }})

    def index_single_image(self, image_path):
        """
        Index a single image in Elasticsearch.

        Args:
            image_path (str): The path to the image to be indexed.

        Returns:
            dict: The Elasticsearch response.
        """
        if not os.path.isfile(image_path):
            raise ValueError(f"The image file '{image_path}' does not exist.")

        image_feature = self.feature_extractor.extract_features(image_path)
        file_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]

        # Index the image data in Elasticsearch
        resp = self.client.index(index=self.index_name, document={
            "image_vector": image_feature,
            "title": file_name_without_extension,
            "image_url": os.path.basename(image_path)
        })

        return resp

    def index_images(self):
        product_images = glob.glob(os.path.join(self.image_folder_path, '*.jpg'))
        for image_path in product_images:
            image_feature = self.feature_extractor.extract_features(image_path)
            file_name_without_extension = os.path.splitext(os.path.basename(image_path))[
                0]  # Get the file name without extension
            print(file_name_without_extension)
            resp = self.client.index(index=self.index_name, document={
                "image_vector": image_feature,
                "title": file_name_without_extension,  # Use the file name without extension as the title
            })
            print(resp)


if __name__ == "__main__":
    image_path = '../user_image.jpg'
    resnet_feature_extractor = FeatureExtractor('ResNet50')
    index_name = 'product_image_search'
    product_images_folder_path = 'C:\\Users\\kyawt\\Documents\\GitHub\\fileserver\\uploads\\productMainImage'

    image_indexer = ImageIndexer(index_name, product_images_folder_path, resnet_feature_extractor)
    image_indexer.create_index()
    image_indexer.index_images()
