from keras_facenet import FaceNet
import cv2
import os
import numpy as np

# Load the FaceNet model


model = FaceNet()

# Define the paths to the dataset of images and the output files for embeddings and labels
# Dataset is made using retinaFace, you can find it on my github for make such dataset
dataset_path = r'face_images'
embeddings_path = 'embeddings.npy'
labels_path = 'labels.npy'


# Initialize empty lists for the embeddings and labels
embeddings = []
labels = []

# Loop over each subdirectory in the dataset path
for subdirectory in os.listdir(dataset_path):
    subdirectory_path = os.path.join(dataset_path, subdirectory)
    # Loop over each image in the subdirectory
    for filename in os.listdir(subdirectory_path):
        image_path = os.path.join(subdirectory_path, filename)

        # Load the image and extract faces
        image = cv2.imread(image_path)
        faces = model.extract(image, threshold=0.95)

        # Loop over each face and append the embedding and label to the lists
        for face in faces:
            try:
                embedding = face['embedding']
                embeddings.append(embedding)
                labels.append(subdirectory)
            except KeyError:
                print("Cannot extract embedding for face")

# Convert the embeddings and labels to numpy arrays and save them to disk
embeddings = np.array(embeddings)
labels = np.array(labels)
np.save(embeddings_path, embeddings)
np.save(labels_path, labels)
