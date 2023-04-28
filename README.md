# Red Wine Quality Classifier using a Neural Network

This project builds a classification model using a neural network to predict the quality of red wine based on its physical and chemical characteristics. The dataset used in this project is the Red Wine Quality dataset available on [Red Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality).

## Project Overview

The project involves the following steps:

- Data preparation: The dataset is loaded into a pandas dataframe and preprocessed to remove any null values, outliers, and unnecessary features. The target variable "quality" is also converted to a binary classification problem by mapping wine with quality greater than or equal to 7 as good and the rest as bad.
- Data exploration: The preprocessed dataset is analyzed using various visualization techniques to gain insights into the relationship between the features and the target variable.
- Feature engineering: The dataset is standardized using the StandardScaler and the features are transformed into a 2D matrix using Principal Component Analysis (PCA) to improve the model's performance.
- Model building: A neural network is built using Keras with two hidden layers, each containing 64 neurons, and a binary cross-entropy loss function. The model is trained on the preprocessed dataset for 100 epochs with a batch size of 32.
- Model evaluation: The model's performance is evaluated on a holdout dataset using various evaluation metrics like accuracy, precision, recall, and F1-score.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Keras
- Matplotlib
- Seaborn
- Jupyter Notebook
- SMOTE

## Installation and Setup

- Clone the repository:

```bash
git clone https://github.com/Gson-glitch/Red-Wine-Quality-Classifier-using-Neural-Networks.git

You can install these dependencies using pip. For example:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow 
