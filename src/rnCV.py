# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler # Or RobustScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, matthews_corrcoef, average_precision_score,
    log_loss # Added log_loss as another potential metric
)
from sklearn.base import clone # To clone estimators

# Helper function to calculate multiple metrics
def calculate_metrics(y_true, y_pred, y_proba):
    """Calculates a dictionary of classification metrics."""
    metrics = {}
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except Exception as e: metrics['accuracy'] = np.nan; print(f"Could not calculate accuracy: {e}")
    try:
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    except Exception as e: metrics['balanced_accuracy'] = np.nan; print(f"Could not calculate balanced_accuracy: {e}")
    try:
        # F1 score (binary default)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e: metrics['f1'] = np.nan; print(f"Could not calculate f1: {e}")
    try:
        # F2 score (emphasizes recall) - calculated manually beta=2
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        if precision + recall == 0:
             metrics['f2'] = 0.0
        else:
             metrics['f2'] = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)
    except Exception as e: metrics['f2'] = np.nan; print(f"Could not calculate f2: {e}")
    try:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    except Exception as e: metrics['precision'] = np.nan; print(f"Could not calculate precision: {e}")
    try:
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0) # Same as sensitivity
        metrics['sensitivity'] = metrics['recall']
    except Exception as e: metrics['recall'] = np.nan; metrics['sensitivity'] = np.nan; print(f"Could not calculate recall/sensitivity: {e}")
    try:
        # Specificity = TN / (TN + FP)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tn + fp == 0:
            metrics['specificity'] = np.nan # Avoid division by zero if no negatives
        else:
            metrics['specificity'] = tn / (tn + fp)
    except Exception as e: metrics['specificity'] = np.nan; print(f"Could not calculate specificity: {e}")
    try:
        # Negative Predictive Value (NPV) = TN / (TN + FN)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if tn + fn == 0:
             metrics['npv'] = np.nan # Avoid division by zero if no predicted negatives
        else:
             metrics['npv'] = tn / (tn + fn)
    except Exception as e: metrics['npv'] = np.nan; print(f"Could not calculate npv: {e}")
    try:
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    except Exception as e: metrics['mcc'] = np.nan; print(f"Could not calculate mcc: {e}")

    # Metrics requiring probabilities (use y_proba[:, 1] for the positive class)
    if y_proba is not None:
      y_proba_pos = y_proba[:, 1]
      try:
          metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
      except Exception as e: metrics['roc_auc'] = np.nan; print(f"Could not calculate roc_auc: {e}")
      try:
          metrics['pr_auc'] = average_precision_score(y_true, y_proba_pos)
      except Exception as e: metrics['pr_auc'] = np.nan; print(f"Could not calculate pr_auc: {e}")
      try:
          metrics['log_loss'] = log_loss(y_true, y_proba)
      except Exception as e: metrics['log_loss'] = np.nan; print(f"Could not calculate log_loss: {e}")
    else:
      metrics['roc_auc'] = np.nan
      metrics['pr_auc'] = np.nan
      metrics['log_loss'] = np.nan

    return metrics


class RepeatedNestedCV:
    """
    Implements Repeated Nested Cross-Validation for classifier comparison
    following OOP principles.

    Includes internal hyperparameter tuning via inner CV and outer loop
    for performance estimation. Handles preprocessing within pipeline.
    """
    def __init__(self, X, y, estimators, param_grids,
                 n_outer_splits=5, n_inner_splits=3, n_repeats=10,
                 inner_cv_metric='roc_auc', # Metric for hyperparameter tuning
                 scaler=StandardScaler, # Allow choosing scaler (StandardScaler/RobustScaler)
                 imputer=SimpleImputer(strategy='median'), # Allow configuring imputer
                 random_state=42):
        """
        Initializes the RepeatedNestedCV instance.

        Args:
            X (pd.DataFrame or np.ndarray): Features dataset.
            y (pd.Series or np.ndarray): Target variable.
            estimators (dict): Dictionary where keys are estimator names (str)
                               and values are estimator objects (e.g., {'svm': SVC()}).
            param_grids (dict): Dictionary where keys are estimator names (matching keys in estimators)
                                and values are parameter grids (dict or list of dicts)
                                for hyperparameter tuning (e.g., {'svm': {'C': [0.1, 1, 10]}}).
            n_outer_splits (int): Number of folds for the outer CV loop (N).
            n_inner_splits (int): Number of folds for the inner CV loop (K).
            n_repeats (int): Number of times to repeat the N-fold outer CV (R).
            inner_cv_metric (str): Scikit-learn scoring string for inner CV optimization
                                   (e.g., 'roc_auc', 'f1', 'balanced_accuracy').
                                   Must be a metric calculable from predict_proba if applicable.
            scaler (TransformerMixin class): Scaler class to use (e.g., StandardScaler, RobustScaler).
            imputer (TransformerMixin object): Imputer object to use.
            random_state (int): Seed for random number generators for reproducibility.
        """
        self.X = X
        self.y = y
        self.estimators = estimators
        self.param_grids = param_grids
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.n_repeats = n_repeats
        self.inner_cv_metric = inner_cv_metric
        self.scaler_class = scaler
        self.imputer = imputer
        self.random_state = random_state

        # Check that estimators and param_grids keys match
        if set(self.estimators.keys()) != set(self.param_grids.keys()):
            raise ValueError("Estimator names in 'estimators' and 'param_grids' must match.")

        # Storage for results
        # List to store dictionaries, each dict representing one outer fold result
        self.results_ = []


    def run(self):
        """Executes the Repeated Nested Cross-Validation process."""
        print(f"Starting Repeated Nested CV ({self.n_repeats} repeats, {self.n_outer_splits} outer folds, {self.n_inner_splits} inner folds)")

        np.random.seed(self.random_state) # Seed for outer loop splits if needed (though StratifiedKFold handles it)
        outer_fold_counter = 0

        for repeat in range(self.n_repeats):
            print(f"\n--- Repeat {repeat + 1}/{self.n_repeats} ---")
            # Stratified K-Fold for outer loop - new splits for each repeat
            outer_cv = StratifiedKFold(n_splits=self.n_outer_splits, shuffle=True,
                                       random_state=self.random_state + repeat) # Vary seed per repeat

            for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(self.X, self.y)):
                outer_fold_counter += 1
                print(f"  Outer Fold {outer_fold + 1}/{self.n_outer_splits} (Overall Fold {outer_fold_counter})")

                # Get outer training and test sets based on indices
                # Ensure using .iloc for positional indexing if X, y are pandas objects
                if isinstance(self.X, (pd.DataFrame, pd.Series)):
                    X_train_outer, X_test_outer = self.X.iloc[train_outer_idx], self.X.iloc[test_outer_idx]
                    y_train_outer, y_test_outer = self.y.iloc[train_outer_idx], self.y.iloc[test_outer_idx]
                else: # Assume numpy arrays
                    X_train_outer, X_test_outer = self.X[train_outer_idx], self.X[test_outer_idx]
                    y_train_outer, y_test_outer = self.y[train_outer_idx], self.y[test_outer_idx]

                # Iterate through each estimator
                for est_name, estimator in self.estimators.items():
                    print(f"    Processing Estimator: {est_name}")

                    # --- Inner Loop for Hyperparameter Tuning ---
                    best_params, best_inner_score = self._inner_loop(
                        X_train_outer, y_train_outer, estimator,
                        self.param_grids[est_name], est_name
                    )
                    print(f"      Best params for {est_name} in fold {outer_fold_counter}: {best_params} (Inner {self.inner_cv_metric}: {best_inner_score:.4f})")

                    # --- Train Final Model for Outer Fold ---
                    # Create pipeline with best hyperparameters
                    final_pipeline = Pipeline([
                        ('imputer', clone(self.imputer)), # Clone to avoid state leakage
                        ('scaler', self.scaler_class()), # Instantiate scaler
                        (est_name, clone(estimator).set_params(**best_params)) # Clone and set best params
                    ])

                    # Fit pipeline on the entire outer training set
                    final_pipeline.fit(X_train_outer, y_train_outer)

                    # --- Evaluate Performance on Outer Test Set ---
                    y_pred_outer = final_pipeline.predict(X_test_outer)
                    y_proba_outer = None
                    if hasattr(final_pipeline, "predict_proba"):
                        try:
                             y_proba_outer = final_pipeline.predict_proba(X_test_outer)
                        except Exception as e:
                             print(f"      Warning: Could not get probabilities for {est_name}. Prob-based metrics will be NaN. Error: {e}")
                    else:
                        print(f"      Warning: Estimator {est_name} does not support predict_proba. Prob-based metrics will be NaN.")


                    # Calculate metrics
                    outer_metrics = calculate_metrics(y_test_outer, y_pred_outer, y_proba_outer)
                    print(f"      Outer test metrics for {est_name}: { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in outer_metrics.items()} }") # Print formatted metrics

                    # Store results
                    result_record = {
                        'repeat': repeat + 1,
                        'outer_fold': outer_fold + 1,
                        'overall_fold': outer_fold_counter,
                        'estimator': est_name,
                        'best_params': best_params,
                        'inner_cv_score': best_inner_score,
                        **outer_metrics # Unpack calculated metrics into the record
                    }
                    self.results_.append(result_record)

        print("\nRepeated Nested Cross-Validation finished.")
        return self.get_results() # Return results as DataFrame


    def _inner_loop(self, X_train_outer, y_train_outer, estimator, param_grid, est_name):
        """Performs the inner cross-validation loop for hyperparameter tuning."""

        # Stratified K-Fold for inner loop
        inner_cv = StratifiedKFold(n_splits=self.n_inner_splits, shuffle=True,
                                   random_state=self.random_state) # Use base seed for inner loop consistency across outer folds? Or vary? Let's vary.
                                   # Using self.random_state + outer_fold_counter ensures different inner splits if outer folds overlap across repeats.

        # Store scores for each parameter combination
        param_scores = {}

        # Iterate through all combinations in the parameter grid
        for params in ParameterGrid(param_grid):
            inner_fold_scores = []

            # Inner K-Fold loop
            for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):

                # Get inner training and validation sets
                if isinstance(X_train_outer, (pd.DataFrame, pd.Series)):
                    X_train_inner, X_val_inner = X_train_outer.iloc[train_inner_idx], X_train_outer.iloc[val_inner_idx]
                    y_train_inner, y_val_inner = y_train_outer.iloc[train_inner_idx], y_train_outer.iloc[val_inner_idx]
                else: # Assume numpy arrays
                    X_train_inner, X_val_inner = X_train_outer[train_inner_idx], X_train_outer[val_inner_idx]
                    y_train_inner, y_val_inner = y_train_outer[train_inner_idx], y_train_outer[val_inner_idx]

                # Create pipeline for this inner fold
                pipeline = Pipeline([
                    ('imputer', clone(self.imputer)),
                    ('scaler', self.scaler_class()),
                    (est_name, clone(estimator).set_params(**params)) # Clone estimator and set current params
                ])

                try:
                    # Fit pipeline on inner training data
                    pipeline.fit(X_train_inner, y_train_inner)

                    # Evaluate based on the specified inner_cv_metric
                    score = self._calculate_inner_score(pipeline, X_val_inner, y_val_inner)
                    inner_fold_scores.append(score)

                except Exception as e:
                    print(f"      Error during inner fold {inner_fold+1} for params {params}: {e}")
                    inner_fold_scores.append(np.nan) # Append NaN if error occurs

            # Calculate average score across inner folds for these params
            avg_score = np.nanmean(inner_fold_scores) # Use nanmean to ignore potential NaN scores
            param_scores[tuple(sorted(params.items()))] = avg_score # Use sorted tuple of items as key

        # Find the best parameters based on the average inner score
        if not param_scores: # Handle case where no scores were calculated
             print("      Warning: No parameter scores calculated in inner loop.")
             # Return default parameters or first set from grid? Let's return empty dict and NaN score.
             best_param_tuple = ()
             best_score = np.nan
        else:
            best_param_tuple = max(param_scores, key=param_scores.get)
            best_score = param_scores[best_param_tuple]

        best_params_dict = dict(best_param_tuple)
        return best_params_dict, best_score


    def _calculate_inner_score(self, pipeline, X_val, y_val):
        """Calculates the score for the inner CV based on the chosen metric."""
        # Metrics requiring probabilities
        prob_metrics = ['roc_auc', 'pr_auc', 'log_loss'] # pr_auc uses average_precision_score

        if self.inner_cv_metric in prob_metrics:
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba_val = pipeline.predict_proba(X_val)
                    if self.inner_cv_metric == 'roc_auc':
                        return roc_auc_score(y_val, y_proba_val[:, 1])
                    elif self.inner_cv_metric == 'pr_auc':
                         return average_precision_score(y_val, y_proba_val[:, 1])
                    elif self.inner_cv_metric == 'log_loss':
                         return -log_loss(y_val, y_proba_val) # Return negative log loss as higher is better
                except Exception as e:
                    print(f"      Warning: Could not calculate probability-based metric {self.inner_cv_metric}. Error: {e}")
                    return np.nan
            else:
                 print(f"      Warning: Estimator does not support predict_proba for metric {self.inner_cv_metric}.")
                 return np.nan
        else: # Metrics requiring predictions
            try:
                y_pred_val = pipeline.predict(X_val)
                if self.inner_cv_metric == 'accuracy':
                    return accuracy_score(y_val, y_pred_val)
                elif self.inner_cv_metric == 'balanced_accuracy':
                    return balanced_accuracy_score(y_val, y_pred_val)
                elif self.inner_cv_metric == 'f1':
                    return f1_score(y_val, y_pred_val, zero_division=0)
                elif self.inner_cv_metric == 'precision':
                    return precision_score(y_val, y_pred_val, zero_division=0)
                elif self.inner_cv_metric == 'recall':
                    return recall_score(y_val, y_pred_val, zero_division=0)
                elif self.inner_cv_metric == 'mcc':
                    return matthews_corrcoef(y_val, y_pred_val)
                # Add other non-probability metrics if needed
                else:
                    print(f"      Warning: Unsupported inner_cv_metric '{self.inner_cv_metric}'. Returning NaN.")
                    return np.nan
            except Exception as e:
                print(f"      Warning: Could not calculate prediction-based metric {self.inner_cv_metric}. Error: {e}")
                return np.nan


    def get_results(self, as_dataframe=True):
        """
        Returns the collected results.

        Args:
            as_dataframe (bool): If True, returns results as a pandas DataFrame.
                                 Otherwise, returns a list of dictionaries.

        Returns:
            pd.DataFrame or list: Collected performance metrics.
        """
        if as_dataframe:
            return pd.DataFrame(self.results_)
        else:
            return self.results_

