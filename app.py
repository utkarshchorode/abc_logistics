%%writefile app.py

import streamlit as st
import joblib
import pandas as pd

# Load the trained model
logi = joblib.load('logi.sav')

# Title of the Streamlit app
st.title('Delivery Delay Prediction')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Input features from the user
delivery_distance = st.number_input('Delivery Distance (e.g., 10.5)', min_value=0.0, value=19.35, format="%.2f")
traffic_congestion = st.slider('Traffic Congestion (1-5, 5 being most congested)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5, 5 being severe)', 1, 5, 2)
delivery_slot = st.slider('Delivery Slot (1-3, 1:Early, 2:Mid, 3:Late)', 1, 3, 2)
driver_experience = st.number_input('Driver Experience (Years)', min_value=0, value=5, step=1)
num_stops = st.number_input('Number of Stops', min_value=0, value=5, step=1)
vehicle_age = st.number_input('Vehicle Age (Years)', min_value=0, value=3, step=1)
road_condition_score = st.slider('Road Condition Score (1-5, 5 being best)', 1, 5, 3)
package_weight = st.number_input('Package Weight (e.g., 5.0)', min_value=0.0, value=10.0, format="%.2f")
fuel_efficiency = st.number_input('Fuel Efficiency (e.g., 15.0)', min_value=0.0, value=15.0, format="%.2f")
warehouse_processing_time = st.number_input('Warehouse Processing Time (Minutes)', min_value=0, value=60, step=1)

# Create a button to make predictions
if st.button('Predict Delivery Delay'):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([[delivery_distance, traffic_congestion, weather_condition,
                                delivery_slot, driver_experience, num_stops, vehicle_age,
                                road_condition_score, package_weight, fuel_efficiency,
                                warehouse_processing_time]],
                              columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                                       'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                                       'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                                       'Warehouse_Processing_Time'])
    
    # Make prediction
    prediction = logi.predict(input_data)[0]
    prediction_proba = logi.predict_proba(input_data)[0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error('Prediction: There WILL be a Delivery Delay')
    else:
        st.success('Prediction: There will be NO Delivery Delay')
    
    st.write(f"Probability of No Delay: {prediction_proba[0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[1]:.2f}")

st.markdown("""
--- 
To run this Streamlit app, save this code as `app.py` and then run `streamlit run app.py` in your terminal.
""")
