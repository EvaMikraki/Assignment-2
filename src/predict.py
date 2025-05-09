# Import necessary libraries
import pandas as pd
import joblib
import argparse
import os
import sys # To exit script on error

# --- Configuration ---
# Define the expected feature columns based on the training data (X from Task 4)
# It's crucial these match the columns used to train the final pipeline
# (after dropping 'id' and 'diagnosis')
EXPECTED_FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
 ]
# Define potential ID column to drop from input data
ID_COLUMN_TO_DROP = 'id'

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions using a saved scikit-learn pipeline.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved .pkl model pipeline file."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input data file (CSV format) for prediction."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the predictions (CSV format)."
    )
    return parser.parse_args()

# --- Main Prediction Logic ---
def main():
    """Loads model, loads data, prepares features, predicts, and saves."""
    args = parse_arguments()

    # 1. Load the Model Pipeline
    print(f"Loading model from: {args.model_path}")
    try:
        model_pipeline = joblib.load(args.model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Load Input Data
    print(f"Loading input data from: {args.data_path}")
    try:
        input_df = pd.read_csv(args.data_path)
        print(f"Input data shape: {input_df.shape}")
    except FileNotFoundError:
        print(f"Error: Input data file not found at {args.data_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input data: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Prepare Features (X_predict)
    print("Preparing features for prediction...")
    # Keep a copy of original data if needed (e.g., to include ID in output)
    output_df = input_df[[ID_COLUMN_TO_DROP]].copy() if ID_COLUMN_TO_DROP in input_df.columns else pd.DataFrame(index=input_df.index)


    # Drop potential ID column if it exists
    if ID_COLUMN_TO_DROP in input_df.columns:
        input_df = input_df.drop(columns=[ID_COLUMN_TO_DROP])
        print(f"Dropped column: '{ID_COLUMN_TO_DROP}'")

    # Ensure all expected feature columns are present
    missing_cols = set(EXPECTED_FEATURES) - set(input_df.columns)
    if missing_cols:
        print(f"Error: Input data is missing expected feature columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    # Ensure only expected feature columns are used and in the correct order
    try:
        X_predict = input_df[EXPECTED_FEATURES]
        print(f"Prepared features shape: {X_predict.shape}")
    except KeyError:
        print("Error: Could not select all expected features. Check column names.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error preparing features: {e}", file=sys.stderr)
        sys.exit(1)


    # 4. Make Predictions
    print("Making predictions...")
    try:
        predictions = model_pipeline.predict(X_predict)
        print("Predictions generated successfully.")
        # Optional: Predict probabilities if needed
        # try:
        #     probabilities = model_pipeline.predict_proba(X_predict)[:, 1] # Probability of positive class
        #     output_df['probability'] = probabilities
        # except AttributeError:
        #     print("Model does not support predict_proba.")
        # except Exception as e:
        #     print(f"Error predicting probabilities: {e}")

    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        # Print shape/dtypes for debugging if prediction fails
        print("Feature data details before prediction failure:", file=sys.stderr)
        print(X_predict.info(), file=sys.stderr)
        sys.exit(1)

    # 5. Save Predictions
    print(f"Saving predictions to: {args.output_path}")
    output_df['prediction'] = predictions
    # Optional: Map numerical predictions back to labels if desired
    # Assuming 0='B', 1='M' from LabelEncoder used in training script
    # label_map = {0: 'B', 1: 'M'}
    # output_df['prediction_label'] = output_df['prediction'].map(label_map)

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        output_df.to_csv(args.output_path, index=False)
        print("Predictions saved successfully.")
    except Exception as e:
        print(f"Error saving predictions: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- Prediction Script Complete ---")

# --- Entry Point ---
if __name__ == "__main__":
    main()

# **How to Use `predict.py`:**

# 1.  Have a CSV file with the new data (e.g., `holdout_data.csv`).
    # This file *must* contain columns with the same names as the 30 features used for training. It can optionally contain an 'id' column.
# 2.  Run from the command line:
    # ```bash
    # python predict.py --model_path ./models/final_lr_elasticnet_model.pkl --data_path ./holdout_data.csv --output_path ./predictions.csv

# 3.  This will create a `predictions.csv` file containing the predictions (and potentially the 'id' if it was present in the input).
