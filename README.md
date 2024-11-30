Apple Stock Price Prediction using Stacked LSTM
Project Overview
This project demonstrates a time series forecasting approach using a Stacked Long Short-Term Memory (LSTM) network to predict the closing stock price of Apple Inc.. The model leverages historical stock price data for learning and prediction. Key steps include stationarity checks, data transformation, hyperparameter tuning, and model evaluation.

Key Features
Data Preprocessing: Stationarity checks, differencing, and normalization of stock price data.
Model Architecture: Stacked LSTM for sequential data learning.
Evaluation Metrics: Comparison of actual and predicted prices using key metrics like MAE, MSE, RMSE, and R².
Hyperparameter Tuning: Optimized using Optuna for best performance.
Visualization: Plots for stationarity, white noise test, and actual vs predicted prices.
Dataset
The dataset used is Apple Inc.'s historical stock prices, containing columns like:

date: Date of the stock price.
close: Closing price of the stock.
Only the close column was used for model training and prediction.

Tools and Technologies
Languages: Python
Libraries:
Data Processing: pandas, numpy, scikit-learn
Visualization: matplotlib
Model Building: tensorflow.keras
Hyperparameter Tuning: optuna
Statistical Tests: statsmodels
Project Workflow
1. Exploratory Data Analysis
Selected relevant columns (date and close) for analysis.
Checked for missing values and handled them appropriately.
2. Stationarity Check
Performed Augmented Dickey-Fuller (ADF) Test:
Null Hypothesis: The data is non-stationary.
Result: Data was non-stationary; applied differencing to make it stationary.
3. White Noise Test
Conducted Ljung-Box Test:
Null Hypothesis: Data is white noise.
Result: Data was not white noise, confirming the presence of meaningful patterns.
4. Normalization
Applied Min-Max Scaling to normalize the data for LSTM model compatibility.
5. Data Preparation
Split the dataset into sequences of 60-time steps for LSTM input.
Divided the data into training (80%) and test (20%) sets.
6. Model Architecture
Built a Stacked LSTM model:
Two LSTM layers with 50 units each and ReLU activation.
Added Dropout (20%) layers to prevent overfitting.
Used a Dense output layer with 1 unit for the prediction.
7. Hyperparameter Tuning
Used Optuna to tune the following parameters:
Number of units in LSTM layers.
Dropout rate.
Learning rate for the optimizer.
Batch size.
8. Training
Trained the model with:
Early Stopping: Monitored validation loss to stop training when no improvement was observed.
Optimized hyperparameters from Optuna.
9. Evaluation
Evaluated the model on training and test datasets using:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R²)
10. Visualization
Stationarity Plot: Differenced data to confirm stationarity.
Training vs Validation Loss: Loss curves over epochs.
Actual vs Predicted Plot: Comparison of actual stock prices with model predictions.

Results:
Evaluation Metrics for Training Data:
Mean Absolute Error (MAE): 1.99
Mean Squared Error (MSE): 7.99
Root Mean Squared Error (RMSE): 2.83
R-squared (R²): 0.99
Evaluation Metrics for Test Data:
Mean Absolute Error (MAE): 5.12
Mean Squared Error (MSE): 55.09
Root Mean Squared Error (RMSE): 7.42
R-squared (R²): 0.97

Visualizations:
Plots confirm the model's ability to capture stock price trends effectively.

Future Work
Extend the model to predict multiple stock prices simultaneously.
Experiment with other deep learning architectures like GRU or Transformer models.
Incorporate external factors like financial news sentiment.
Acknowledgments
Special thanks to Optuna and TensorFlow for providing robust tools for hyperparameter optimization and deep learning.


