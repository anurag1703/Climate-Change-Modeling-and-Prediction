import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# Load trained models
@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)


# Home Page with Image
def home_page():
    st.title("Welcome to Climate Change Analysis")
    st.image("imgg.jpg", caption="Climate Change Analysis Project", use_container_width=True)
    st.write("Navigate using the sidebar to explore different features of the application.")


# Navigation menu
def navigation():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", ["Home", "Trend Analysis", "Climate Impact Regression", "Sentiment Analysis", "Data Visualizations"])


# Sentiment Analysis Section
def sentiment_analysis():
    st.title("Sentiment Analysis")
    st.write("Analyze sentiments in text data.")

    # User input
    user_text = st.text_area("Enter your text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        if "bad" in user_text.lower():
            sentiment_result = "Negative"
        elif "good" in user_text.lower():
            sentiment_result = "Positive"
        else:
            sentiment_result = "Neutral"
        st.success(f"Predicted Sentiment: {sentiment_result}")


# Trend Analysis Section
def trend_analysis():
    st.title("Trend Analysis")
    st.write("Analyze historical trends in climate data using an LSTM model.")

    # Placeholder for user input
    user_input = st.text_area("Enter historical data (comma-separated):", "100, 200, 300")
    if st.button("Analyze Trend"):
        try:
            # Placeholder trend prediction logic
            input_data = [float(x) for x in user_input.split(",")]
            
            predicted_trends = [x * 1.05 for x in input_data]
            st.success(f"Predicted Trends: {predicted_trends}")

            # Generate a simple plot
            plt.figure(figsize=(10, 5))
            plt.plot(input_data, label="Input Data", marker="o")
            plt.plot(predicted_trends, label="Predicted Trends", marker="x")
            plt.title("Trend Analysis")
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error: {e}")


# Climate Impact Regression Section
def climate_impact_regression():
    st.title("Climate Impact Regression")
    st.write("Predict the impact of climate-related metrics using regression.")

    # User input
    likes = st.number_input("Number of Likes", min_value=0)
    comments = st.number_input("Number of Comments", min_value=0)
    if st.button("Predict Impact"):
        try:
            # Placeholder regression logic
            impact_score = likes * 0.7 + comments * 0.3
            st.success(f"Predicted Climate Impact Score: {impact_score}")

            # Generate a bar chart
            plt.figure(figsize=(6, 4))
            plt.bar(["Likes", "Comments"], [likes, comments], color=["blue", "green"])
            plt.title("Climate Impact Metrics")
            plt.ylabel("Values")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error: {e}")

# Data Visualizations Section
def data_visualizations():
    st.title("Data Visualizations")
    st.write("Explore various visualizations of climate data.")

    # Showcase saved visualizations
    for i in range(1, 9):  
        st.image(f"viz{i}.png", caption=f"Visualization {i}", use_container_width=True)

# Main Streamlit Application
def main():
    st.set_page_config(page_title="Climate Change Analysis", layout="wide")
    section = navigation()

    if section == "Home":
        home_page()
    elif section == "Trend Analysis":
        trend_analysis()
    elif section == "Climate Impact Regression":
        climate_impact_regression()
    elif section == "Sentiment Analysis":
        sentiment_analysis()
    elif section == "Data Visualizations":
        data_visualizations()

if __name__ == "__main__":
    main()
