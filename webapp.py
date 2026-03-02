from flask import Flask, jsonify
import numpy as np

app = Flask(__name__)

# Simulated data
data = list(np.random.rand(50))

@app.route("/api/data")
def get_data():
    return jsonify({"data": data[-50:]})

if __name__ == "__main__":
    app.run(port=5000)
