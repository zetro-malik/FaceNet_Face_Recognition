from keras_facenet import FaceNet
import cv2
import numpy as np

# Load the FaceNet model


model = FaceNet()

embeddings_path = 'embeddings.npy'
labels_path = 'labels.npy'


# Load the embeddings and labels from disk
known_embeddings = np.load(embeddings_path)
known_labels = np.load(labels_path)


image = cv2.imread(
    r"test.jpg")

# Extract faces from the image and calculate embeddings
faces = model.extract(image, threshold=0.50)
test_embeddings = np.array([face['embedding'] for face in faces])

# Define a threshold distance for matching faces
threshold_distance = 0.95

# Loop over each test embedding and find the closest matching known embedding
for i, test_embedding in enumerate(test_embeddings):
    distances = np.linalg.norm(known_embeddings - test_embedding, axis=1)
    closest_match_index = np.argmin(distances)
    closest_match_distance = distances[closest_match_index]
    closest_match_label = known_labels[closest_match_index]

    # Draw a bounding box and label for the recognized face
    face_box = faces[i]['box']
    x, y, w, h = face_box[0], face_box[1], face_box[2], face_box[3]

    if closest_match_distance < threshold_distance:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, closest_match_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, '?', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 10)

# Display the output image
cv2.imwrite('detected.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
