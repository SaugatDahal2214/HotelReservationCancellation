import pickle
import numpy as np

# Load the saved model
with open("ml_model", "rb") as f:
    model = pickle.load(f)

# New sample data for prediction as a 2D array
new_data = np.array([
   [2,2,0,2,1,0,5,114,5,20,1,0,0,0,198.9,0,1]
])

# Confirm the shape of new_data
print("Shape of new_data:", new_data.shape)  # Should be (1, number_of_features)

# Make predictions
predictions = model.predict(new_data, n_features=model.n_features)

# Print predictions
print("Predictions:", predictions)
