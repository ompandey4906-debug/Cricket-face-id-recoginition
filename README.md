# 🏏 Indian Cricket Team Member Face Identification

A computer vision project that identifies selected Indian cricket players from a facial image. The application detects a face, extracts facial features, predicts the most likely player using a Support Vector Machine (SVM) classifier, and can reject an image when the detected face does not sufficiently match a known player.

> **Prototype:** https://ai.studio/apps/5741875d-e30a-4cae-879a-8e2b417db6ff?fullscreenApplet=true
>
> <img width="1880" height="907" alt="image" src="https://github.com/user-attachments/assets/92f27912-d9c3-4f90-ad9a-a899ff6a2c70" />

<img width="1916" height="852" alt="image" src="https://github.com/user-attachments/assets/796d062a-6dde-428c-babf-e95ffda95e1a" />

<img width="1896" height="872" alt="image" src="https://github.com/user-attachments/assets/29e5e956-d348-4009-81c0-a1e1a210d281" />




## ✨ Features

- Upload a player image in JPG, JPEG, or PNG format.
- Detect and crop a face using MTCNN.
- Extract a 512-dimensional facial embedding using a pretrained FaceNet/InceptionResnetV1 model.
- Classify the embedding using an SVM model.
- Apply a cosine-similarity threshold to help flag unfamiliar faces as `UNKNOWN`.

## 🧠 How It Works

```text
Input Image
    ↓
Face Detection and Cropping (MTCNN)
    ↓
Facial Feature Extraction (FaceNet / InceptionResnetV1)
    ↓
512-Dimensional Embedding
    ↓
SVM Classification
    ↓
Similarity-Based Unknown-Face Check
    ↓
Predicted Player Name or UNKNOWN
```

## 🛠️ Tech Stack

- **Python** — application and inference logic
- **PyTorch** — deep-learning inference
- **facenet-pytorch** — MTCNN and InceptionResnetV1
- **scikit-learn** — SVM classification and similarity utilities
- **NumPy** — embedding and label arrays
- **Pillow** — image loading and conversion
- **joblib** — loading the saved SVM classifier
- **Streamlit** — web interface

## 👥 Supported Player Classes

The reference implementation lists these ten classes:

- Virat Kohli
- Rohit Sharma
- Jasprit Bumrah
- Hardik Pandya
- Ravindra Jadeja
- Rishabh Pant
- KL Rahul
- Shubman Gill
- Mohammed Siraj
- Suryakumar Yadav

The supported classes depend on the labels contained in the trained model artifacts. Update the list if the model is retrained with a different dataset.

## 📁 Project Structure

```text
Indian_Cricket_Face_ID/
├── app.py
├── requirements.txt
├── src/
│   └── predict.py
└── models/
    ├── cricket_face_svm.pkl
    ├── training_embeddings.npy
    ├── training_labels.npy
    └── evaluation_results.txt
```

The model files must be present in the `models/` directory for the reference inference code to run. The pretrained FaceNet weights are loaded by `facenet-pytorch` when required and are not the same as the SVM and NumPy artifacts listed above.

## 🗂️ Dataset

The reference implementation identifies the [Indian Cricketers Dataset on Kaggle](https://www.kaggle.com/datasets/aviratgupta/indian-cricketers-dataset) as its source. Review its current contents, permitted use, labels, and quality before using it.

For a new training pipeline, organize images by person, for example:

```text
dataset/
├── Virat_Kohli/
│   ├── image_001.jpg
│   └── image_002.jpg
├── Rohit_Sharma/
│   ├── image_001.jpg
│   └── image_002.jpg
└── ...
```

Use a separate, held-out test set to evaluate the model. Avoid placing near-duplicate images of the same photograph in both training and test sets, as this can inflate evaluation scores.

**Training note:** The reference inference layout loads saved model artifacts; a compatible training script is needed to regenerate those artifacts for a new dataset or new player classes. Do not reuse old embeddings or an old SVM after changing the embedding model, preprocessing, or class labels.

## 📊 Evaluation

The reference repository's `models/evaluation_results.txt` reports the following metrics for its original run:

- Classes: 10
- Training samples: 200
- Testing samples: 50
- Embedding size: 512
- Accuracy: 94.00%
- Precision: 95.00%
- Recall: 94.00%
- F1-score: 93.66%

These are **reported reference results, not a guarantee of performance on your machine, your own dataset, or a newly trained model**. Regenerate evaluation metrics after retraining, and assess false matches as well as correct predictions.


## 📜 Credits and Data Use

- Reference implementation: [Indian_Cricket_Face_ID on GitHub](https://github.com/vaani240/Indian_Cricket_Face_ID)
- Dataset reference: [Indian Cricketers Dataset on Kaggle](https://www.kaggle.com/datasets/aviratgupta/indian-cricketers-dataset)
- Face models and libraries: [facenet-pytorch](https://github.com/timesler/facenet-pytorch), [PyTorch](https://pytorch.org/), and [scikit-learn](https://scikit-learn.org/)

