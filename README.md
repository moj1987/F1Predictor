# 🏎️ F1 Race Pace Predictor

## What is this?
F1Predictor is a Streamlit web application that analyzes Formula 1 Free Practice 2 (FP2) telemetry data to discover the true race pace of every team. It extracts long-run averages, calculates tire degradation rates, and feeds everything into an advanced, dynamic Machine Learning pipeline to predict the final race classification.

## The ML Architecture
The predictive model has been completely engineered to think exactly like a real F1 Strategist:

1. **The 15-Race Rolling Brain**: The `RandomForestRegressor` trains dynamically on the last 15 historical races before every prediction. This natively teaches it the most recent car regulations and performance physics. 
2. **Era Boundaries**: Driver and Team momentum calculations are strictly fenced to the current season to prevent contamination from previous generations of cars.
3. **Track Affinity & Sponsor Immunity**: We calculate each driver's historical average finish at the exact circuit over the last 3 years. The matching algorithm uses `Country` and `Location` logic to safely bypass yearly F1 sponsor name changes.
4. **Intelligent Rookie Fallbacks**: If a rookie has never raced at a track, the model intelligently falls back to their Current Season Momentum, but applies a mathematically sound `+ 4.0` penalty to simulate their lack of track experience compared to veterans.
5. **Target Variable Engineering (F1 Points)**: The ML model is trained to predict **F1 Points** (25, 18, 15... 0) instead of Race Positions (1-20). By doing this, the algorithm stops wasting processing power trying to rank the bottom 10 drivers and redirects 100% of its mathematical optimization to accurately ranking the podium!

## How to Run

1. **Install Dependencies**
   Make sure you have Python installed, then install the required packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Web App**
   The Machine Learning model trains dynamically at runtime using the FastF1 API! You do not need to pre-train any static model files. Just launch the interface:
   ````bash
   streamlit run app.py
   ````