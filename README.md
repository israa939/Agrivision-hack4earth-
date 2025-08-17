# AgriVision – Automated Farming Robot


## Project Overview
**AgriVision** is an automated farming robot designed to optimize agricultural practices through precision farming and machine learning. It collects real-time farm data and predicts plant diseases early, helping farmers take preventive measures to reduce crop loss and improve productivity.

---

## Objective
The primary goal of this project is to **predict plant diseases** using machine learning models. By analyzing environmental data and crop images, the robot provides timely alerts for healthier crops and sustainable farming.

---

## Key Features

- **Real-time Data Monitoring:** Measures soil moisture, temperature, and other crop health indicators.
- **Disease Prediction:** Uses a **Convolutional Neural Network (CNN)** to classify plant diseases from images.
- **Kaggle Dataset Integration:** Downloads and analyzes datasets from Kaggle for training and validation.
- **Autonomous Operations:** Handles essential farming tasks, reducing manual labor.

---

## Technologies Used

- **Programming Language:** Python  
- **Machine Learning Frameworks:** TensorFlow, Keras  
- **Data Handling & Visualization:** Pandas, NumPy, Matplotlib  
- **Model Evaluation:** Scikit-learn  
- **Development Environment:** Google Colab  
- **Dataset Access:** Kaggle API  

---

## CNN Model Architecture

- **Convolutional Layers:** Learn spatial hierarchies from crop images.  
- **Max Pooling Layers:** Reduce feature dimensionality.  
- **Dropout Layers:** Prevent overfitting.  
- **Dense Layers:** Perform final classification of healthy vs diseased plants.  

The CNN is trained on labeled crop image datasets for accurate disease detection.

---

## Getting Started

1. **Download the Project:**  
   - Click the green **Code** button on GitHub and select **Download ZIP**.  
   - Extract the ZIP file on your computer.

2. **Download the Dataset:**  
   - Go to the [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download)  
   - Download the dataset and extract it.  
   - Place the dataset folder inside your project directory (e.g., `AgriVision/dataset`).

3. **Open in Google Colab:**  
   - Go to [Google Colab](https://colab.research.google.com/).  
   - Click **File → Upload notebook** or **File → Open notebook → Upload** to open your `.ipynb` file from the extracted folder.

4. **Run the Notebook:**  
   - Make sure all dependencies are installed (Colab usually has Python libraries like TensorFlow, Pandas, and NumPy pre-installed).  
   - Run the notebook cells to train and test the model.



