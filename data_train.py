import glob
import json
import os

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

# Load a pre-trained deep learning model (VGG16 in this case)
base_model = VGG16(weights='imagenet', include_top=False)


# Define a function to extract features from an image
def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features


# Example: Extract features from product images in your database
# Specify the folder path where your product images are located
product_images_folder_path = 'C:\\Users\\kyawt\\Documents\\GitHub\\fileserver\\uploads\\productAllImage'
product_images = glob.glob(os.path.join(product_images_folder_path, '*.jpg'))
print(len(product_images))
product_features = [{"features": extract_features(image_path, base_model).tolist(), "product": image_path} for
                    image_path in product_images]

# Define the path to your JSON file
json_file_path = 'training_data.json'

# Serialize the training data dictionary to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(product_features, json_file)

print("Training data has been saved to", json_file_path)
