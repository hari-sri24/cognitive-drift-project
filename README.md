# Cognitive Drift Detection System for Real-Time Decision Pattern Analysis

## 1. Project Objective
The project detects sudden changes (cognitive drift) in decision patterns in real-time and visualizes them on a live dashboard.  
It uses a **Flask backend** to generate data and detect drift, and a **Streamlit dashboard** to show the results interactively.

---

## 2. Project Structure

Cognitive-Drift-Project/
│
├── backend_api.py # Flask backend that generates data and detects drift
├── dashboard_app.py # Streamlit dashboard for real-time visualization
├── drift_detection.py # Drift detection logic using statistical tests
├── data/ # Folder to store generated CSV data
├── requirements.txt # Python dependencies
└── README.md # Project explanation

---

## 3. How to Run

### Step 1: Install dependencies

pip install -r requirements.txt
cd C:\Intership\Cognitive-Drift-Project

# Step 2: Start the Flask backend #

cd C:\Intership\Cognitive-Drift-Project

# Step 3: Start the Streamlit dashboard

python -m streamlit run dashboard_app.py

# Step 4: Open in browser

Dashboard: http://localhost:8501

Backend API (JSON data): http://127.0.0.1:5000/api/data

---

## 4. Output / Features

Live Graph: Shows real-time decision values
Drift Status: ✅ Stable or ⚠ Drift Detected
P-value Metric: Shows statistical significance of change
Real-time Update: Refreshes every 3 seconds
Cumulative Data: Can show historical trend (if implemented)

# 5. How it Works
Backend (Flask) generates decision data continuously and checks for drift.
Drift Detection uses statistical methods (Kolmogorov–Smirnov Test) to detect pattern changes.
Streamlit Dashboard fetches API data and visualizes:
Line chart of decision values
Drift status metric
P-value metric


## 6. Future Improvements

Add ML model to predict drift automatically
Make dashboard cumulative for long-term trends
Add user authentication for multi-user monitoring
Store historical data for analytics and reporting

7. References

Streamlit Documentation: https://docs.streamlit.io
Flask Documentation: https://flask.palletsprojects.com
Kolmogorov–Smirnov Test: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test



---

 **What to do now:**  

1. Open VS Code → project folder  
2. Create a new file → **`README.md`**  
3. Paste this content → save  

> This **alone makes your project look professional** for internship submission or viva.  

---

If you want, I can **also make a small diagram** of Backend → Drift Detection → Dashboard and **add it to the README**.  
It will make your report **look very impressive**.  

Do you want me to do that next?
