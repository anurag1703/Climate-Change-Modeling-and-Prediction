# Climate Change Modeling Dashboard

This project is a comprehensive Streamlit dashboard designed to analyze climate change data and provide actionable insights through three core functionalities:

1. **Trend Prediction**: Use an LSTM model to predict future trends in climate data.
2. **Regression**: Predict comments count using a regression model based on features like likes count.
3. **Sentiment Analysis**: Analyze public sentiment about climate change using a multiclass RNN classifier.
4. **Data Visualization**: Display climate data in a visually appealing manner to facilitate understanding and exploration.
---

## Features

### Navigation
The app includes a sidebar for seamless navigation between the following sections:

- **Home**: Introduction and navigation guide.
- **Trend Prediction**: Predict trends based on input features.
- **Regression**: Perform regression-based predictions.
- **Sentiment Analysis**: Analyze sentiment from textual data.
- **Data Visualization**: Visualize climate data for better understanding.

### Models Used
- **LSTM for Trend Prediction**
- **Linear Regression for Comments Prediction**
- **RNN (LSTM-based) for Sentiment Analysis**

---

## Installation

### Prerequisites
- Python 3.12+
- Streamlit
- Required libraries listed in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Usage

### Trend Prediction
1. Navigate to the "Trend Prediction" section.
2. Input a list of dates (comma-separated).
3. Click "Predict Trend" to get LSTM predictions for future trends.

### Regression
1. Navigate to the "Regression" section.
2. Input the likes count and other relevant details.
3. Click "Predict Comments Count" to see the regression model's output.

### Sentiment Analysis
1. Navigate to the "Sentiment Analysis" section.
2. Enter a textual input describing climate change.
3. Click "Analyze Sentiment" to see the predicted sentiment.


---

## Folder Structure
```
.
├── app.py                # Streamlit app
├── models                # Trained models directory
├── data                  # Data directory
├── scripts               # Additional Python scripts
├── requirements.txt      # Required libraries
└── README.md             # Project documentation
```

---

