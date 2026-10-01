import pandas as pd
import numpy as np
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

# Read the CSV file into a pandas DataFrame
data = pd.read_csv("multiple.csv")


# --------------------------------------------------
# 2. Select independent variables
# --------------------------------------------------

# X contains the input features used to predict Exam_Score
X = data[
    [
        "Study_Hours",
        "Sleep_Hours",
        "Attendance_Percentage",
        "Previous_Score"
    ]
]


# --------------------------------------------------
# 3. Select dependent variable
# --------------------------------------------------

# y contains the value that we want to predict
y = data["Exam_Score"]


# --------------------------------------------------
# 4. Split data into training and testing sets
# --------------------------------------------------

# 80% data is used for training
# 20% data is used for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 5. Start MLflow experiment
# --------------------------------------------------

# Give a name to our MLflow experiment
mlflow.set_experiment("Exam Score Prediction")


# Start one MLflow run
# Everything logged inside this block will be saved by MLflow
with mlflow.start_run():

    # --------------------------------------------------
    # 6. Log parameters
    # --------------------------------------------------

    # Store the model name in MLflow
    mlflow.log_param("model", "Linear Regression")

    # Store the test data percentage
    mlflow.log_param("test_size_percentage", 20)

    # Store the number of training records
    mlflow.log_param("train_size", len(X_train))

    # Store the number of testing records
    mlflow.log_param("test_size", len(X_test))


    # --------------------------------------------------
    # 7. Create the model
    # --------------------------------------------------

    # Create a Linear Regression model
    model = LinearRegression()


    # --------------------------------------------------
    # 8. Train the model
    # --------------------------------------------------

    # The model learns the relationship between X_train and y_train
    model.fit(X_train, y_train)


    # --------------------------------------------------
    # 9. Make predictions
    # --------------------------------------------------

    # Predict Exam_Score using the test data
    y_pred = model.predict(X_test)


    # --------------------------------------------------
    # 10. Calculate evaluation metrics
    # --------------------------------------------------

    # MSE measures the average squared prediction error
    mse = mean_squared_error(y_test, y_pred)

    # RMSE is the square root of MSE
    rmse = np.sqrt(mse)

    # R² shows how well the model explains the target variable
    r2 = r2_score(y_test, y_pred)


    # --------------------------------------------------
    # 11. Log metrics into MLflow
    # --------------------------------------------------

    # Save MSE in MLflow
    mlflow.log_metric("MSE", mse)

    # Save RMSE in MLflow
    mlflow.log_metric("RMSE", rmse)

    # Save R² Score in MLflow
    mlflow.log_metric("R2_Score", r2)


    # --------------------------------------------------
    # 12. Print results
    # --------------------------------------------------

    print("Train Size:", len(X_train))
    print("Test Size:", len(X_test))

    print("\nMean Squared Error:", mse)
    print("RMSE:", rmse)
    print("R² Score:", r2)


# --------------------------------------------------
# 13. Predict score for a new student
# --------------------------------------------------

# Create input data for a new student
new_student = pd.DataFrame({
    "Study_Hours": [7],
    "Sleep_Hours": [8],
    "Attendance_Percentage": [90],
    "Previous_Score": [72]
})


# Use the trained model to predict the student's exam score
predicted_score = model.predict(new_student)


# Display the prediction
print("\nPredicted Exam Score:", predicted_score[0])