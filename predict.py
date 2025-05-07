
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Sample input (you can modify this)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)
print("Predicted class:", prediction[0])
