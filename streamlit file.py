import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the trained model and preprocessing steps
model = joblib.load('car_price_prediction_pipeline.pkl')


# Load dataset for filtering and identifying similar data
d = pd.read_csv('C:\\Users\\91959\\PycharmProjects\\pythonProject5\\cardheko_data.csv')

# Set pandas option to handle future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)
# Features used for training


# Define features and target
features = ['it', 'ft', 'bt', 'modelYear', 'transmission', 'oem', 'City', 'Seats', 'mileage', 'km']
X = d[features]
y = d['price']

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Create column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ]), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ]
)


# Function to filter data based on user selections
def filter_data(oem=None,  body_type=None, fuel_type=None, seats=None,it=None,modelYear=None):
    filtered_data = d.copy()
    if oem:
        filtered_data = filtered_data[filtered_data['oem'] == oem]
    if body_type:
        filtered_data = filtered_data[filtered_data['bt'] == body_type]
    if fuel_type:
        filtered_data = filtered_data[filtered_data['ft'] == fuel_type]
    if seats:
        filtered_data = filtered_data[filtered_data['Seats'] == seats]
    if it:
        filtered_data = filtered_data[filtered_data['it'] == seats]
    if modelYear:
        filtered_data = filtered_data[filtered_data['modelYear'] ==modelYear ]
    return filtered_data


# Preprocessing function for user input

    # Apply label encoding


# Streamlit Application
st.title("Car Price Prediction")

# Sidebar for user inputs
st.sidebar.header('Input Car Features')

# Set background colors
input_background_color = "lightblue"  
result_background_color = "pink"

st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container {{
        background-color: {result_background_color};
    }}
    .stButton>button {{
        background-color: lightblue;
        color: white;
    }}
    .result-container {{
        text-align: center;
        background-color: {result_background_color};
        padding: 20px;
        border-radius: 10px;
    }}
    .prediction-title {{
        font-size: 28px;
        color: maroon;
    }}
    .prediction-value {{
        font-size: 36px;
        font-weight: bold;
        color: maroon;
    }}
    .info {{
        font-size: 18px;
        color: grey;
    }}
    .sidebar .sidebar-content {{
        background-color: {input_background_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Get user inputs with visual representation
def visual_selectbox(label, options, index=0):
    selected_option = st.sidebar.selectbox(label, options, index=index)
    return selected_option


# Get user inputs in a defined order
selected_oem = visual_selectbox('1. Original Equipment Manufacturer (OEM)', d['oem'].unique())
filtered_data = filter_data(oem=selected_oem)

body_type = visual_selectbox('2. Body Type', filtered_data['bt'].unique())
filtered_data = filter_data(oem=selected_oem,  body_type=body_type)

fuel_type = visual_selectbox('3. Fuel Type', filtered_data['ft'].unique())
filtered_data = filter_data(oem=selected_oem,  body_type=body_type, fuel_type=fuel_type)

transmission = visual_selectbox('4. Transmission Type', filtered_data['transmission'].unique())
filtered_data = filter_data(oem=selected_oem,  body_type=body_type, fuel_type=fuel_type,)

seat_count = visual_selectbox('5. Seats', filtered_data['Seats'].unique())
filtered_data = filter_data(oem=selected_oem,  body_type=body_type, fuel_type=fuel_type,
                            seats=seat_count)

it = visual_selectbox('6. Ignition type', filtered_data['it'].unique())
filtered_data = filter_data(oem=selected_oem,  body_type=body_type, fuel_type=fuel_type,
                            seats=seat_count,it=it)
modelYear = visual_selectbox('7. Model Year', filtered_data['modelYear'].unique())
filtered_data = filter_data(oem=selected_oem,  body_type=body_type, fuel_type=fuel_type,
                            seats=seat_count,it=it,modelYear=modelYear)

km = st.sidebar.number_input('8. Kilometers Driven', min_value=0, max_value=500000, value=10000)

# Adjust mileage slider
min_mileage = np.floor(filtered_data['mileage'].min())
max_mileage = np.ceil(filtered_data['mileage'].max())

# Ensure mileage slider has an interval of 0.5
min_mileage = float(min_mileage)
max_mileage = float(max_mileage)

mileage = st.sidebar.slider('9. Mileage (kmpl)', min_value=min_mileage, max_value=max_mileage, value=min_mileage,
                            step=0.5)

city = visual_selectbox('10. City', d['City'].unique())

# Create a DataFrame for user input
user_input_data = {
    'ft': [fuel_type],
    'bt': [body_type],
    'km': [km],
    'it':[it],
    'transmission': [transmission],
    'oem': [selected_oem],
    'modelYear': [modelYear],
    'City': [city],
    'mileage': [mileage],
    'Seats': [seat_count]

}

user_df = pd.DataFrame(user_input_data)

# Ensure the columns are in the correct order and match the trained model's features
user_df = user_df[features]

# Button to trigger prediction
if st.sidebar.button('Predict'):
    if user_df.notnull().all().all():
        try:
            # Make prediction
            prediction = model.predict(user_df)

            st.markdown(f"""
                <div class="result-container">
                    <h2 class="prediction-title">Predicted Car Price</h2>
                    <p class="prediction-value">₹{prediction[0]:,.2f}</p>

                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        missing_fields = [col for col in user_df.columns if user_df[col].isnull().any()]
        st.error(f"Missing fields: {', '.join(missing_fields)}. Please fill all required fields.")
