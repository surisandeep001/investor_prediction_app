# investment_prediction.py

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Set up Streamlit app
st.set_page_config(page_title="Investor Type Prediction", layout="wide")

# Define the number of records (investors)
num_investors = 5000  # Increase the number of records for better training

# Generate random investor data
investor_ids = range(1, num_investors + 1)
ages = np.random.randint(18, 81, size=num_investors)
investment_amounts = np.random.uniform(5, 500, size=num_investors)
tenures = np.random.randint(1, 30, size=num_investors)
num_funds = np.random.randint(1, 20, size=num_investors)

# Define classification logic based on investment behavior
def classify_investor(age, investment_amount, tenure, num_funds):
    if investment_amount > 300 and num_funds > 10 and tenure > 10:
        return 'Aggressive'
    elif investment_amount > 100 and num_funds >= 5 and tenure >= 5:
        return 'Moderate'
    else:
        return 'Conservative'

# Apply classification logic to create the 'Investment Type' label
investment_types = [
    classify_investor(age, invest_amt, tenure, funds)
    for age, invest_amt, tenure, funds in zip(ages, investment_amounts, tenures, num_funds)
]

# Create the DataFrame
df = pd.DataFrame({
    'Investor ID': investor_ids,
    'Age': ages,
    'Investment Amount (in 1000s)': investment_amounts,
    'Tenure (years)': tenures,
    'Number of Funds': num_funds,
    'Investment Type': investment_types
})

# Encode the Investment Type labels
label_encoder = LabelEncoder()
df['Investment Type Encoded'] = label_encoder.fit_transform(df['Investment Type'])

# Prepare features and target variable
X = df[['Age', 'Investment Amount (in 1000s)', 'Tenure (years)', 'Number of Funds']]
y = df['Investment Type Encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set up hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

grid_search = GridSearchCV(estimator=XGBClassifier(random_state=42), param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

# Train the model
grid_search.fit(X_train, y_train)

# Get the best model
final_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#st.write(f"Model accuracy on test set: {accuracy:.2f}")

# Add a banner using Markdown with HTML
st.markdown(
    """
    <div style="background-color: #4CAF50; padding: 10px; border-radius: 5px;">
        <h1 style="color: white;">Investor Type Prediction App by COMPOUNDEXPRESS</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Title and Introduction
st.markdown("""This app predicts the investment type of an individual based on their profile. Enter the details below to see your predicted investment type!""")

# Input form for user data with better UI
st.header("Input Investor Profile")

age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)
investment = st.number_input("Investment Amount (in 1000s)", min_value=5, max_value=500, value=100)
tenure = st.slider("Tenure (years)", min_value=1, max_value=30, value=5, step=1)
num_funds = st.number_input("Number of Funds", min_value=1, max_value=20, value=3)

# When the "Predict" button is clicked
if st.button("Predict"):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([[age, investment, tenure, num_funds]], 
                              columns=['Age', 'Investment Amount (in 1000s)', 'Tenure (years)', 'Number of Funds'])
    
    # Scale input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = final_model.predict(input_data_scaled)
    
    # Convert prediction to label
    predicted_label = label_encoder.inverse_transform(prediction)
    
    # Show the prediction with a stylish output
    st.subheader("Prediction Result")

    # Define a color for each investment type
    colors = {
        'Aggressive': 'red',
        'Moderate': 'orange',
        'Conservative': 'blue'
    }

    # Create the formatted output
    output = f"<span style='font-size: 24px; font-weight: bold; color: {colors[predicted_label[0]]};'>{predicted_label[0]}</span>"
    st.markdown(f"**Predicted Investor Type:** {output}", unsafe_allow_html=True)

    # Provide additional context with different colors
    if predicted_label[0] == 'Aggressive':
        st.markdown("<span style='color: red;'>You have an aggressive investment strategy, which typically involves higher risks but also higher potential returns.</span>", unsafe_allow_html=True)
    elif predicted_label[0] == 'Moderate':
        st.markdown("<span style='color: orange;'>You have a moderate investment strategy, balancing risk and reward effectively.</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color: blue;'>You have a conservative investment strategy, focusing on preserving capital with lower risks.</span>", unsafe_allow_html=True)

# Display the app layout options and help section
st.sidebar.title("Help Section")
st.sidebar.info("This application predicts the investment type based on your provided details. Adjust the sliders and inputs accordingly.")
