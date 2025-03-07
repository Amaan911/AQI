import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Admin\Desktop\city_day.csv")  # Update path if needed
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Handle missing values
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
    return df

df = load_data()

# Streamlit App Title
st.title("🌍 Air Quality Index (AQI) Prediction & Analysis")

# Sidebar
st.sidebar.header("Air Quality Prediction App")
st.sidebar.markdown("Adjust parameters below:")

# User Input in Sidebar
selected_city = st.sidebar.selectbox("Select a City", df['City'].unique())

# Filter data for selected city
filtered_df = df[df["City"] == selected_city]

# User-defined AQI parameters
pm25 = st.sidebar.slider("PM2.5 Level", float(filtered_df["PM2.5"].min()), float(filtered_df["PM2.5"].max()), float(filtered_df["PM2.5"].mean()))
pm10 = st.sidebar.slider("PM10 Level", float(filtered_df["PM10"].min()), float(filtered_df["PM10"].max()), float(filtered_df["PM10"].mean()))
no2 = st.sidebar.slider("NO2 Level", float(filtered_df["NO2"].min()), float(filtered_df["NO2"].max()), float(filtered_df["NO2"].mean()))
so2 = st.sidebar.slider("SO2 Level", float(filtered_df["SO2"].min()), float(filtered_df["SO2"].max()), float(filtered_df["SO2"].mean()))
o3 = st.sidebar.slider("O3 Level", float(filtered_df["O3"].min()), float(filtered_df["O3"].max()), float(filtered_df["O3"].mean()))

# Dataset Overview
st.subheader(f"📊 Data Overview for {selected_city}")
st.dataframe(filtered_df.head())  # Show first 5 rows of selected city

# Dataset Statistics
st.subheader("📈 Dataset Statistics")
st.table(filtered_df.describe())

# Define Features & Target Variable
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'Year', 'Month', 'Day']
filtered_df = filtered_df.dropna(subset=['AQI'])
X = filtered_df[features].dropna()
y = filtered_df.loc[X.index, 'AQI']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance Metrics")
st.metric(label="Mean Absolute Error", value=round(mae, 2))
st.metric(label="Mean Squared Error", value=round(mse, 2))
st.metric(label="R2 Score", value=round(r2, 2))

# AQI Prediction Function
def predict_aqi(pm25, pm10, no2, so2, o3, year):
    input_data = np.array([[pm25, pm10, 0, no2, 0, 0, 0, so2, o3, 0, 0, 0, year, 1, 1]])
    return model.predict(input_data)[0]

aqi_prediction = predict_aqi(pm25, pm10, no2, so2, o3, 2024)

# Display Prediction
st.subheader("🔮 Predicted AQI")
st.metric(label="Air Quality Index", value=round(aqi_prediction, 2))

# Future AQI Prediction
st.subheader("🚀 Future AQI Prediction")
years_to_predict = st.slider("Select years to predict", 1, 10, 5)  # User input for years

future_years = list(range(2024, 2024 + years_to_predict))
future_aqi = [predict_aqi(pm25, pm10, no2, so2, o3, year) for year in future_years]

# Show Future Prediction
future_df = pd.DataFrame({"Year": future_years, "Predicted AQI": future_aqi})
st.write(future_df)

# Future AQI Visualization
st.subheader("📈 Predicted AQI Trends")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=future_years, y=future_aqi, marker="o", ax=ax)
ax.set_title("Future AQI Prediction")
ax.set_xlabel("Year")
ax.set_ylabel("Predicted AQI")
st.pyplot(fig)

# 🔹 **Global Graphs (For Full Dataset)** 🔹

# 1️⃣ AQI Trends Over Time (Whole Dataset)
st.subheader("📉 AQI Trends Over Time (Global Data)")
fig = px.line(df, x='Date', y='AQI', title="Overall AQI Trends Over Time", labels={'AQI': 'Air Quality Index'})
st.plotly_chart(fig)

# 2️⃣ Average AQI Per City
st.subheader("🏙️ Average AQI Per City")
city_avg_aqi = df.groupby('City')['AQI'].mean().sort_values()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=city_avg_aqi.index, y=city_avg_aqi.values, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

# 🔹 **Dynamic Graphs (Based on City & Parameters)** 🔹

# 3️⃣ AQI Trends Over Time for Selected City
st.subheader(f"📉 AQI Trends in {selected_city}")
fig = px.line(filtered_df, x='Date', y='AQI', title=f"AQI Trends in {selected_city}", labels={'AQI': 'Air Quality Index'})
st.plotly_chart(fig)

# 4️⃣ Pollutant Levels Changing with Parameters
st.subheader(f"📊 Pollutant Levels in {selected_city} (Based on Adjusted Parameters)")
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']
pollutant_values = [pm25, pm10, no2, so2, o3]

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=pollutants, y=pollutant_values, ax=ax)
ax.set_title(f"Pollutant Levels in {selected_city} (Dynamic)")
st.pyplot(fig)

st.sidebar.text("Developed by Amaan Ahmamd")
