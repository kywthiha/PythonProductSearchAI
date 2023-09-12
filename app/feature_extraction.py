import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image


class FeatureExtractor:
    def __init__(self, model_name):
        if model_name == 'ResNet50':
            self.base_model = ResNet50(weights='imagenet', include_top=False)
            self.preprocess_input = resnet_preprocess_input
        elif model_name == 'VGG16':
            self.base_model = VGG16(weights='imagenet', include_top=False)
            self.preprocess_input = vgg_preprocess_input
        else:
            raise ValueError("Unsupported model name. Choose 'ResNet50' or 'VGG16'.")

    def extract_features(self, image_path):
        """
        Extracts features from an image using the specified model.

        Args:
            image_path (str): The path to the image.

        Returns:
            list: A list of feature values extracted from the image.
        """
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = self.preprocess_input(img_array)

        # Extract features from the model
        features = self.base_model.predict(img_array)

        # Add a Global Average Pooling layer to obtain a dense vector
        gap = GlobalAveragePooling2D()(features)

        return gap.numpy()[0]


# Example usage for ResNet50
if __name__ == "__main__":
    image_path = '../user_image.jpg'
    resnet_feature_extractor = FeatureExtractor('ResNet50')
    extracted_features = resnet_feature_extractor.extract_features(image_path)
    print("ResNet50 Extracted Features:", extracted_features)
    print("ResNet50 Extracted Features Dims:", len(extracted_features))

# Example usage for VGG16
if __name__ == "__main__":
    image_path = '../user_image.jpg'
    vgg_feature_extractor = FeatureExtractor('VGG16')
    extracted_features = vgg_feature_extractor.extract_features(image_path)
    print("VGG16 Extracted Features:", extracted_features)
    print("VGG16 Extracted Features Dims:", len(extracted_features))
