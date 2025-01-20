# Heart Disease Prediction Project - Setup Guide

## Project Overview
This project aims to predict the likelihood of heart disease in patients based on clinical and lifestyle features. It includes extensive Exploratory Data Analysis (EDA), data visualization, and the development of a machine learning model using the Random Forest algorithm. The dataset used for this project contains anonymized patient records with features such as age, cholesterol levels, blood pressure, and more. The model is designed to assist healthcare professionals in identifying at-risk individuals efficiently.

---

## Prerequisites

Before you begin, ensure you have the following software and tools installed:

- **Python 3.8 or later**
- **Package Manager:** pip or conda
- **Libraries:**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyterlab (optional, for running notebooks)

---

## Installation Instructions

Follow these steps to set up the project:

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   ```

2. **Navigate to the project directory:**
   ```bash
   cd <repository_name>
   ```

3. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv env
   source env/bin/activate    # On Windows: .\env\Scripts\activate
   ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Preparation

1. **Download the dataset:**
   - The dataset can be downloaded from [this link](<dataset_download_link>).
   - Save the dataset as `heart_disease.csv` in the `data/` directory within the project folder.

2. **Preprocess the data:**
   - The data preprocessing script is included in the project.
   - Run the following command to preprocess the dataset:
     ```bash
     python preprocess.py
     ```
   - This script will handle missing values, encode categorical features, and split the data into training and testing sets.

---

## Running the Project

1. **Perform EDA and Visualization:**
   - Open the Jupyter Notebook:
     ```bash
     jupyter lab
     ```
   - Navigate to the `eda_visualization.ipynb` notebook and run all cells.

2. **Train the Random Forest Model:**
   - Execute the training script:
     ```bash
     python train_model.py
     ```

3. **Make Predictions:**
   - Use the model to make predictions on new data:
     ```bash
     python predict.py --input <input_file.csv>
     ```

---

## Model Evaluation

1. **Evaluate the Model:**
   - After training, the model evaluation results will be saved in the `results/` directory.
   - Metrics include:
     - **Accuracy:** Proportion of correctly predicted cases.
     - **Precision, Recall, F1-Score:** For understanding the balance between false positives and false negatives.
     - **Confusion Matrix:** Visualizes true/false positives and negatives.

2. **View Evaluation Metrics:**
   - Open the `evaluation_results.txt` file in the `results/` directory to review detailed metrics.

---

## Troubleshooting

- **Issue:** Python command not found.
  - **Solution:** Ensure Python is correctly installed and added to your system PATH.

- **Issue:** Module not found error during script execution.
  - **Solution:** Ensure all dependencies are installed using `pip install -r requirements.txt`.

- **Issue:** Dataset not found.
  - **Solution:** Verify that the dataset is downloaded and placed in the `data/` directory with the correct name (`heart_disease.csv`).

- **Issue:** Jupyter Notebook not opening.
  - **Solution:** Ensure JupyterLab is installed: `pip install jupyterlab`.

---

## Notes for Advanced Users
- For hyperparameter tuning, modify the configuration file `config.json`.
- The project includes logging; check the `logs/` directory for detailed run logs.
- To use a different machine learning algorithm, replace the `RandomForestClassifier` in `train_model.py` with your desired model.

---

This setup guide ensures that users can replicate the project environment, perform EDA, train the model, and evaluate its performance. If you encounter issues, please raise them in the project repository.
