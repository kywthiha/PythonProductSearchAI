import glob
import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image

# Load a pre-trained deep learning model (VGG16 in this case)
base_model = VGG16(weights='imagenet', include_top=False)

# Define the path to your JSON file
json_file_path = 'training_data.json'  # Replace with the actual path

# Load the training data from the JSON file
with open(json_file_path, 'r') as json_file:
    training_data = json.load(json_file)
product_features = [np.array(item["features"]) for item in training_data]

# Define a function to extract features from an image
def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features


# Example: Extract features from a user-uploaded image
user_image_path = 'user_image.jpg'
user_features = extract_features(user_image_path, base_model)

# Example: Extract features from product images in your database
# Specify the folder path where your product images are located
product_images_folder_path = 'C:\\Users\\kyawt\\Documents\\GitHub\\fileserver\\uploads\\\productMainImage'
product_images = glob.glob(os.path.join(product_images_folder_path, '*.jpg'))

# Calculate cosine similarity between the user's image and product images
similarities = [
    cosine_similarity(user_features.reshape(1, -1), pf.reshape(1, -1))[0][0] if pf.shape == user_features.shape else 0
    for pf in
    product_features]

product_ranking = list(sorted(zip(similarities, product_images), reverse=True))

# Sort product images by similarity
sorted_products = [{"score": score, "product": product} for score, product in
                   sorted(zip(similarities, product_images), reverse=True)]

# Get the top N similar products (e.g., top 10)
top_similar_products = sorted_products[:10]  # Replace 10 with your desired number

# Print the sorted list of product images based on similarity
print("Top 10 Similar products:")
for product in top_similar_products:
    print(product['score'], os.path.basename(product['product']))
