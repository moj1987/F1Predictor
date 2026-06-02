# 🏎️ F1 Race Pace Predictor

## What is this?
F1Predictor is a Streamlit web application that analyzes Formula 1 Free Practice 2 (FP2) telemetry data to discover the true race pace of every team. It filters out slow laps to calculate the average pace by tire compound and uses a basic Machine Learning model to predict the final race classification.

## How to Run

1. **Install Dependencies**
   Make sure you have Python installed, then install the required packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model (If you haven't already)**
   Before generating predictions in the app, you need to train the base model (which will create `dumb_model.pkl`):
   *(Note: Add the specific python command to run your training script here, e.g., `python model.py` or a dedicated training script).*

3. **Start the Web App**
   Launch the Streamlit interface by running:
   ```bash
   streamlit run app.py
   ```
   This will automatically open the application in your default web browser!