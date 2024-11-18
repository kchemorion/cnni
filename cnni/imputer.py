import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CNNImputer:
    def __init__(
        self, 
        max_k: int = 10,
        temporal_weight: float = 0.3,
        spatial_weight: float = 0.3,
        categorical_weight: float = 0.4,
        distance_metric: str = 'euclidean'
    ):
        """
        Initialize the Contextual Nearest Neighbor Imputation (CNNI) model.
        
        Args:
            max_k: Maximum number of neighbors to consider
            temporal_weight: Weight for temporal context
            spatial_weight: Weight for spatial context
            categorical_weight: Weight for categorical context
            distance_metric: Distance metric for nearest neighbors
        """
        self.max_k = max_k
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight
        self.categorical_weight = categorical_weight
        self.distance_metric = distance_metric
        
    def _preprocess_data(self, data):
        """Preprocess input data."""
        if isinstance(data, pd.DataFrame):
            # Store column names for later
            self.column_names = data.columns
            
            # Initialize scaler if not already done
            if not hasattr(self, 'scaler'):
                self.scaler = StandardScaler()
                self.scaler.fit(data)
            
            # Transform the data
            data = self.scaler.transform(data)
        return data.astype(float)
    
    def _inverse_transform(self, data):
        """Inverse transform scaled data."""
        if hasattr(self, 'scaler'):
            data = self.scaler.inverse_transform(data)
        return data
    
    def _calculate_context_weights(self, data, temporal_info=None, spatial_info=None, categorical_info=None):
        """Calculate weights based on context."""
        n_samples = len(data)
        weights = np.ones((n_samples, n_samples))
        
        if temporal_info is not None:
            temporal_info = temporal_info.to_numpy().reshape(-1, 1)
            temporal_dist = np.abs(temporal_info - temporal_info.T)
            temporal_weights = 1 / (1 + temporal_dist)
            weights *= self.temporal_weight * temporal_weights
        
        if spatial_info is not None:
            spatial_info = spatial_info.astype(float)
            spatial_dist = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                spatial_dist[i] = np.sqrt(np.sum((spatial_info - spatial_info[i]) ** 2, axis=1))
            spatial_weights = 1 / (1 + spatial_dist)
            weights *= self.spatial_weight * spatial_weights
        
        if categorical_info is not None:
            categorical_info = categorical_info.astype(float)
            categorical_dist = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                categorical_dist[i] = np.sum(categorical_info != categorical_info[i], axis=1)
            categorical_weights = 1 / (1 + categorical_dist)
            weights *= self.categorical_weight * categorical_weights
        
        return weights
    
    def _get_adaptive_k(self, data, missing_mask):
        """Determine adaptive k based on data density."""
        n_samples = len(data)
        n_features = data.shape[1]
        missing_ratio = missing_mask.sum() / (n_samples * n_features)
        
        # Adjust k based on missing ratio and data size
        k = min(
            self.max_k,
            max(3, int(n_samples * (1 - missing_ratio) * 0.1))
        )
        return k
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        temporal_info: Optional[pd.Series] = None,
        spatial_info: Optional[pd.DataFrame] = None,
        categorical_info: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fit the imputer and transform the input data.
        
        Args:
            X: Input DataFrame with missing values
            temporal_info: Optional temporal information
            spatial_info: Optional spatial information
            categorical_info: Optional categorical information
            
        Returns:
            Imputed DataFrame
        """
        # Preprocess data
        data = self._preprocess_data(X)
        missing_mask = np.isnan(data)
        
        if not np.any(missing_mask):
            return X
        
        # Convert temporal_info to numpy array if needed
        if temporal_info is not None:
            temporal_info = np.array(temporal_info)
        
        # Initialize output array
        imputed_data = data.copy()
        
        # Process each feature separately to save memory
        for feature_idx in range(data.shape[1]):
            feature_missing = missing_mask[:, feature_idx]
            if not np.any(feature_missing):
                continue
            
            feature_data = data[:, feature_idx]
            feature_known = ~feature_missing
            
            if not np.any(feature_known):
                print(f"Warning: All values missing in feature {feature_idx}")
                continue
            
            # Process missing values in batches
            batch_size = 1000  # Adjust based on available memory
            missing_indices = np.where(feature_missing)[0]
            known_indices = np.where(feature_known)[0]
            
            print(f"Processing feature {feature_idx}: {len(missing_indices)} missing values")
            
            for batch_start in range(0, len(missing_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(missing_indices))
                batch_indices = missing_indices[batch_start:batch_end]
                
                # Calculate weights for this batch
                batch_weights = np.zeros((len(batch_indices), len(known_indices)))
                
                # Calculate temporal weights if available
                if temporal_info is not None:
                    temporal_batch = temporal_info[batch_indices].reshape(-1, 1)
                    temporal_known = temporal_info[known_indices].reshape(1, -1)
                    temporal_dist = np.abs(temporal_batch - temporal_known)
                    batch_weights += self.temporal_weight * (1 / (1 + temporal_dist))
                
                # Calculate spatial weights if available
                if spatial_info is not None:
                    spatial_batch = spatial_info[batch_indices]
                    spatial_known = spatial_info[known_indices]
                    for i in range(len(batch_indices)):
                        spatial_dist = np.sqrt(np.sum((spatial_known - spatial_batch[i]) ** 2, axis=1))
                        batch_weights[i] += self.spatial_weight * (1 / (1 + spatial_dist))
                
                # Calculate categorical weights if available
                if categorical_info is not None:
                    categorical_batch = categorical_info[batch_indices]
                    categorical_known = categorical_info[known_indices]
                    for i in range(len(batch_indices)):
                        categorical_dist = np.sum(categorical_known != categorical_batch[i], axis=1)
                        batch_weights[i] += self.categorical_weight * (1 / (1 + categorical_dist))
                
                # Normalize weights
                row_sums = batch_weights.sum(axis=1)
                if np.any(row_sums == 0):
                    # If any row has all zero weights, use equal weights
                    zero_rows = row_sums == 0
                    batch_weights[zero_rows] = 1 / len(known_indices)
                else:
                    batch_weights /= row_sums[:, np.newaxis]
                
                # Find k nearest neighbors
                k = min(self.max_k, len(known_indices))
                neighbor_indices = np.argsort(-batch_weights, axis=1)[:, :k]
                
                # Get known values
                known_values = feature_data[known_indices]
                
                # Calculate weighted average
                for i, missing_idx in enumerate(batch_indices):
                    neighbors = known_values[neighbor_indices[i]]
                    neighbor_weights = batch_weights[i, neighbor_indices[i]]
                    if np.sum(neighbor_weights) == 0:
                        # If all weights are zero, use mean of neighbors
                        imputed_data[missing_idx, feature_idx] = np.mean(neighbors)
                    else:
                        # Renormalize weights
                        neighbor_weights = neighbor_weights / np.sum(neighbor_weights)
                        imputed_data[missing_idx, feature_idx] = np.average(
                            neighbors, weights=neighbor_weights
                        )
                
                if batch_end % 1000 == 0:
                    print(f"  Processed {batch_end}/{len(missing_indices)} values")
        
        # Convert back to DataFrame
        imputed_data = pd.DataFrame(self._inverse_transform(imputed_data), columns=self.column_names)
        
        return imputed_data
    
    def evaluate(
        self,
        X_true: pd.DataFrame,
        X_imputed: pd.DataFrame,
        mask: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Evaluate imputation performance.
        
        Args:
            X_true: True data
            X_imputed: Imputed data
            mask: Boolean mask indicating missing values
            
        Returns:
            Tuple of (RMSE, MAE)
        """
        # Extract values where mask is True (imputed values)
        true_values = []
        imputed_values = []
        
        for col in X_true.columns:
            col_mask = mask[:, X_true.columns.get_loc(col)]
            if col_mask.any():
                true_vals = X_true[col][col_mask].values
                imputed_vals = X_imputed[col][col_mask].values
                
                # Only include non-NaN values
                valid_mask = ~(np.isnan(true_vals) | np.isnan(imputed_vals))
                if valid_mask.any():
                    true_values.extend(true_vals[valid_mask])
                    imputed_values.extend(imputed_vals[valid_mask])
        
        # Convert to numpy arrays
        true_values = np.array(true_values)
        imputed_values = np.array(imputed_values)
        
        if len(true_values) == 0:
            return 0.0, 0.0
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
        mae = mean_absolute_error(true_values, imputed_values)
        
        return rmse, mae
    
    def plot_results(
        self,
        X_true: pd.DataFrame,
        X_imputed: pd.DataFrame,
        mask: pd.DataFrame,
        n_features: int = 3
    ) -> None:
        """
        Plot imputation results.
        
        Args:
            X_true: True data
            X_imputed: Imputed data
            mask: Boolean mask indicating missing values
            n_features: Number of features to plot
        """
        n_features = min(n_features, len(X_true.columns))
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
        if n_features == 1:
            axes = [axes]
            
        for i, col in enumerate(X_true.columns[:n_features]):
            ax = axes[i]
            
            # Get mask for current column
            col_mask = mask[:, X_true.columns.get_loc(col)]
            
            if col_mask.any():
                # Plot true vs imputed values
                sns.scatterplot(
                    x=X_true[col][col_mask],
                    y=X_imputed[col][col_mask],
                    ax=ax,
                    alpha=0.5
                )
                
                # Add perfect prediction line
                min_val = min(
                    X_true[col][col_mask].min(),
                    X_imputed[col][col_mask].min()
                )
                max_val = max(
                    X_true[col][col_mask].max(),
                    X_imputed[col][col_mask].max()
                )
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                
                ax.set_xlabel('True Values')
                ax.set_ylabel('Imputed Values')
                ax.set_title(f'True vs Imputed Values for {col}')
                ax.legend()
            else:
                ax.text(
                    0.5, 0.5,
                    'No imputed values for this feature',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes
                )
            
        plt.tight_layout()
        plt.show()
