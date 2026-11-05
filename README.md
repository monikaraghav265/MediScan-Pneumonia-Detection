# 🩺 MediScan – Chest X-ray Pneumonia Detection

MediScan is a machine learning-based web application that classifies chest X-ray images into three categories: **Normal, Bacterial Pneumonia, and Viral Pneumonia**.

Built using Python, Scikit-learn, OpenCV, and Streamlit, the application allows users to upload chest X-ray images and view model predictions.

## 🚀 Features

- Upload chest X-ray images in JPG, JPEG, or PNG format.
- Predict Normal, Bacterial Pneumonia, or Viral Pneumonia.
- Symptom checker based on selected symptoms.
- Generate and download PDF reports.
- Simple and interactive web interface using Streamlit.

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- OpenCV
- Pillow
- Joblib
- FPDF

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression
- Image preprocessing: Grayscale conversion and resizing to 64 × 64 pixels
- Feature scaling: StandardScaler
- Training dataset: 600 images (200 images per class)
- Validation accuracy: **77.5%**

The model was evaluated using an 80:20 training-validation split.

## 📂 Project Structure

```text
MediScan-Pneumonia-Detection/
│
├── app.py
├── train_and_save_model.py
├── pneumonia_model.pkl
├── requirements.txt
├── report.pdf
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/monikaraghav265/MediScan-Pneumonia-Detection.git
```

**2. Navigate to the project folder**

```bash
cd MediScan-Pneumonia-Detection
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

The application will open in your web browser.

## 📊 Model Training

To retrain the model, arrange the training images in the following structure:

```text
train/
├── NORMAL/
├── BACTERIAL/
└── VIRAL/
```

Then run:

```bash
python train_and_save_model.py
```

The trained model will be saved as `pneumonia_model.pkl`.

The training dataset is not included in this repository.

## ⚠️ Disclaimer

MediScan is an educational machine learning project developed for learning and demonstration purposes. Its predictions and symptom checker are not medically validated and must not be used for actual diagnosis or treatment decisions. Always consult a qualified healthcare professional.

## 👩‍💻 Author

**Monika Raghav**

GitHub: [@monikaragh
