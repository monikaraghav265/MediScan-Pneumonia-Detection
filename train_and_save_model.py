
import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

data_dir = "train"

label_map = {
    "NORMAL": 0,
    "BACTERIAL": 1,
    "VIRAL": 2
}

def load_images_and_labels(data_dir, image_size=(64, 64)):
    X = []
    y = []

    for label, label_id in label_map.items():
        folder = os.path.join(data_dir, label)

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")

        for img_name in os.listdir(folder)[:200]:
            img_path = os.path.join(folder, img_name)

            try:
                img = Image.open(img_path).convert("L")
                img = img.resize(image_size)
                X.append(np.array(img).flatten())
                y.append(label_id)
            except Exception:
                continue

    return np.array(X), np.array(y)

X, y = load_images_and_labels(data_dir)

print("Total images:", len(X))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:", accuracy)

dump(model, "pneumonia_model.pkl")

print("Model saved successfully!")
