<p align="center">
  <img src="hamoye.webp" alt="HamoyeLogo" width="200">
</p>

# Project Title: Electricity Demand and Supply Gap Prediction Model

## Overview
This project aims to predict electricity access and demand gaps using a neural network model, deployed as a web application through Flask and Render. The project involves several key stages: data collection, preprocessing, feature engineering, model training, evaluation, and deployment, providing an interactive interface for users to input data and receive predictions.

## Project Structure
- *data/*: Contains the dataset used for training the model.
- *app.py*: The Flask application that serves the model and handles user requests.
- *templates/*: Contains HTML files for the user interface.
  - *index.html*: The main page where users can input data and see predictions.
- *static/*: Contains static files like CSS and JavaScript (if any).
- *model/*: Contains the trained model files (if applicable).

## Data Collection
The dataset was collected from reliable sources providing data on various factors affecting electricity access and demand, including urban and rural electricity access, population metrics, financial support, energy sources, and socio-economic indicators.

## Data Preparation
1. *Data Loading*: The dataset is loaded, and key features are selected for the prediction model.
2. *Data Cleaning*: Missing values in the selected features are handled to ensure the dataset is complete and ready for modeling.
3. *Feature Scaling*: The data is standardized using StandardScaler to ensure that all features contribute equally to the model's performance.
4. *Train-Test Split*: The data is split into training and testing sets using train_test_split to evaluate the model's performance on unseen data.

## Feature Engineering
1. *Feature Selection*: Relevant features impacting electricity access and demand are selected based on domain knowledge and exploratory data analysis.
2. *Encoding Categorical Features*: Categorical features are converted to numerical representations using one-hot encoding (if applicable).
3. *Handling Target Variable*: The target variable, gap, is transformed to ensure non-negative values if negative gaps do not make sense in the context.

## Model Training
1. *Model Definition*: 
   - An LSTM (Long Short-Term Memory) model is defined for capturing temporal dependencies in the data.
   - An alternative Feedforward Neural Network (FNN) model with dense layers and dropout regularization is also defined using TensorFlow and Keras.
2. *Model Compilation*: The model is compiled with the Adam optimizer and mean squared error loss function.
3. *Model Training*: The model is trained on the training set and validated on the test set for 100 epochs to ensure robustness and accuracy.
4. *Model Evaluation*: The model is evaluated on training and test sets to ensure it generalizes well to new data.
5. *Model Prediction*: The model makes predictions on both training and test sets, combining true values and predicted values for comprehensive evaluation.
6. *Performance Metrics*: The root mean squared error (RMSE) and mean absolute error (MAE) are calculated for detailed model performance assessment.

## Deployment
1. *Flask Application*: A Flask application is created to serve the model and handle user inputs.
2. *HTML Template*: An HTML file (index.html) is designed to provide a user interface for input and output.
3. *Render Deployment*: 
   - The application is pushed to GitHub.
   - A Render account is created, and a new web service is set up.
   - The GitHub repository is linked to Render.
   - Environment variables are configured, and the service is deployed, making the application live.

    

## Usage
1. Open a web browser and go to : https://elec-deploy.onrender.com/predict



## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Thanks to the data providers for the comprehensive dataset.
- Special thanks to the stakeholders for their continuous support and feedback.
