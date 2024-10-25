import pandas as pd
import numpy as np
from decision_tree import DecisionTree  # Adjust the import based on your file structure

print("Script started...")

        
def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print(data.head())
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return data, X, y

def main():
    print("Starting the training process...")
    file_path = '../data/fProcessed.csv'
    data, X, y = load_data(file_path)

    if X is None or y is None:
        print("Data loading failed. Exiting...")
        return

    print("Data loaded. Proceeding with train-test split...")
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print(f"Training on {len(X_train)} samples and testing on {len(X_test)} samples.")

    num_features = X_train.shape[1]
    feature_names = data.columns[:-1].tolist()
    tree = DecisionTree(n_features=num_features, feature_names=feature_names, min_samples_split=2, max_depth=10)
    
    print("Fitting the model...")
    tree.fit(X_train, y_train)
    print("Model training completed.")

    print("Making predictions...")
    predictions = tree.predict(X_test, feature_names)
    
    print("Predictions made. Calculating accuracy...")
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()
