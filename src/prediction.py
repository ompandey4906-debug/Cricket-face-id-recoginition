
import os
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import joblib


# Project root = folder containing app.py
BASE_PATH = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_PATH,
    "models",
    "cricket_face_svm.pkl"
)

EMBEDDINGS_PATH = os.path.join(
    BASE_PATH,
    "models",
    "training_embeddings.npy"
)

LABELS_PATH = os.path.join(
    BASE_PATH,
    "models",
    "training_labels.npy"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



svm_model = joblib.load(MODEL_PATH)


mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40,
    keep_all=False,
    device=device
)



resnet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)


X_train_saved = np.load(
    EMBEDDINGS_PATH
)

y_train_saved = np.load(
    LABELS_PATH,
    allow_pickle=True
)


def create_centroids():

    centroids = {}

    for player in svm_model.classes_:

        embeddings = X_train_saved[
            y_train_saved == player
        ]

        centroid = np.mean(
            embeddings,
            axis=0
        )

        centroid = centroid / np.linalg.norm(
            centroid
        )

        centroids[player] = centroid

    return centroids


def predict_player(
    image_path,
    threshold=0.60
):

    image = Image.open(
        image_path
    ).convert("RGB")


    # Detect and crop face
    face = mtcnn(image)


    if face is None:

        return {
            "prediction": "No face detected",
            "similarity": 0.0
        }


    # Prepare tensor
    face_tensor = face.unsqueeze(
        0
    ).to(device)


    # Generate FaceNet embedding
    with torch.no_grad():

        embedding = resnet(
            face_tensor
        ).cpu().numpy()[0]


    svm_prediction = svm_model.predict(
        embedding.reshape(1, -1)
    )[0]


    centroids = create_centroids()

    centroid = centroids[
        svm_prediction
    ]


    similarity = cosine_similarity(
        embedding.reshape(1, -1),
        centroid.reshape(1, -1)
    )[0][0]


    if similarity < threshold:

        prediction = "UNKNOWN"

    else:

        prediction = svm_prediction


    return {
        "prediction": prediction,
        "similarity": float(similarity),
        "closest_player": svm_prediction
    }