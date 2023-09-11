import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D

# Load a pre-trained deep learning model (VGG16 in this case)
base_model = VGG16(weights='imagenet', include_top=False)


# Define a function to extract features from an image
def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    # Add a Global Average Pooling layer to obtain a dense vector
    gap = GlobalAveragePooling2D()(features)

    return gap.numpy().flatten().tolist()
