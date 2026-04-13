## 📌 Overview
This project focuses on predicting irrigation needs (Low, Medium, High) using machine learning and deep learning techniques.

The goal is to optimize agricultural water usage by leveraging environmental and soil features such as temperature, moisture, rainfall, and sunlight.

---

##Dataset
The dataset includes:
- Temperature (°C)
- Soil Moisture
- Rainfall (mm)
- Sunlight Hours
- Previous Irrigation
- Other environmental features

Target variable:
- Irrigation_Need → {Low, Medium, High}

---

## Project Workflow

### 1. Data Preprocessing
- Removed unnecessary columns (`id`, `Water_Source`, `Irrigation_Type`)
- Checked missing values and duplicates
- Converted target labels to numerical format

### 2. Feature Engineering
Created new features to improve model performance:
- `water_stress` = Temperature / Soil Moisture
- `evaporation` = Temperature × Sunlight Hours
- `total_water` = Rainfall + Previous Irrigation

### 3. Encoding
- Applied **Ordinal Encoding** to categorical features

### 4. Scaling
- Used **StandardScaler** for normalization

### 5. Model Building (Deep Learning)
- Neural Network using Keras:
  - Dense layers (ReLU activation)
  - Output layer (Softmax)
- Loss: `sparse_categorical_crossentropy`
- Optimizer: Adam

### 6. Handling Imbalanced Data
- Used **class weights** to balance training

### 7. Training
- EarlyStopping to prevent overfitting
- Validation split for monitoring performance

### 8. Evaluation
- Accuracy and loss tracking
- Visualization of training history

### 9. Prediction
- Generated predictions for test dataset
- Converted predictions back to labels (Low, Medium, High)

---

## Model Architecture
- Dense (265 units, ReLU)
- Dense (265 units, ReLU)
- Dense (128 units, ReLU)
- Output Layer (3 classes, Softmax)

---

## Results
The model successfully learns patterns in environmental data and predicts irrigation levels effectively, with improvements from feature engineering and class balancing giving accuracy > 97%
---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/irrigation-prediction.git
