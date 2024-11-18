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
        self.scaler = StandardScaler()
        
    def _preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data."""
        # Create a copy to avoid modifying the original data
        X_processed = X.copy()
        
        # Identify numeric columns
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        
        # Scale numeric features
        if len(numeric_cols) > 0:
            X_processed[numeric_cols] = self.scaler.fit_transform(X_processed[numeric_cols])
            
        return X_processed
        
    def _calculate_context_weights(
        self, 
        X: pd.DataFrame,
        temporal_info: Optional[pd.Series] = None,
        spatial_info: Optional[pd.DataFrame] = None,
        categorical_info: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Calculate context-aware weights for each data point."""
        n_samples = len(X)
        weights = np.ones((n_samples, n_samples))
        
        # Apply temporal weighting if temporal information is provided
        if temporal_info is not None:
            temporal_dist = np.abs(temporal_info.values.reshape(-1, 1) - temporal_info.values.reshape(1, -1))
            temporal_weights = np.exp(-self.temporal_weight * temporal_dist)
            weights *= temporal_weights
            
        # Apply spatial weighting if spatial information is provided
        if spatial_info is not None:
            spatial_dist = np.sqrt(
                np.sum(
                    (spatial_info.values.reshape(n_samples, 1, -1) - 
                     spatial_info.values.reshape(1, n_samples, -1)) ** 2,
                    axis=2
                )
            )
            spatial_weights = np.exp(-self.spatial_weight * spatial_dist)
            weights *= spatial_weights
            
        # Apply categorical weighting if categorical information is provided
        if categorical_info is not None:
            categorical_dist = (categorical_info.values.reshape(n_samples, 1, -1) != 
                              categorical_info.values.reshape(1, n_samples, -1)).mean(axis=2)
            categorical_weights = np.exp(-self.categorical_weight * categorical_dist)
            weights *= categorical_weights
            
        return weights
    
    def _get_adaptive_k(self, X: pd.DataFrame, point_idx: int) -> int:
        """Determine adaptive k based on local data density."""
        # Simple density-based adaptation
        density = np.sum(~X.isna().iloc[point_idx])
        k = max(2, min(self.max_k, int(np.sqrt(density))))
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
        X_processed = self._preprocess_data(X)
        
        # Calculate context weights
        context_weights = self._calculate_context_weights(
            X_processed, temporal_info, spatial_info, categorical_info
        )
        
        # Create output DataFrame
        X_imputed = X_processed.copy()
        
        # Iterate through columns with missing values
        for col in X_processed.columns[X_processed.isna().any()]:
            # Get indices of missing and non-missing values
            missing_mask = X_processed[col].isna()
            valid_mask = ~missing_mask
            
            if sum(valid_mask) == 0:
                continue
            
            # Get non-missing data for the current column
            valid_data = X_processed[valid_mask].fillna(X_processed[valid_mask].mean())
            missing_data = X_processed[missing_mask].fillna(X_processed[valid_mask].mean())
            
            # Initialize and fit nearest neighbors on non-missing data
            nbrs = NearestNeighbors(metric=self.distance_metric)
            nbrs.fit(valid_data)
            
            # Find k nearest neighbors for each missing value
            missing_indices = np.where(missing_mask)[0]
            for idx in missing_indices:
                # Get adaptive k for this point
                k = self._get_adaptive_k(X_processed, idx)
                
                # Get the data point with missing value
                query_point = X_processed.iloc[[idx]].fillna(X_processed[valid_mask].mean())
                
                # Find k nearest neighbors
                distances, indices = nbrs.kneighbors(
                    query_point,
                    n_neighbors=min(k, sum(valid_mask))
                )
                
                # Get the actual indices of valid data points
                valid_indices = np.where(valid_mask)[0]
                neighbor_indices = valid_indices[indices[0]]
                
                # Apply context weights
                weights = context_weights[idx, neighbor_indices]
                weights = weights / weights.sum()
                
                # Compute weighted average for imputation
                X_imputed.iloc[idx, X_imputed.columns.get_loc(col)] = np.average(
                    X_processed.iloc[neighbor_indices, X_processed.columns.get_loc(col)],
                    weights=weights
                )
        
        # Inverse transform scaled features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_imputed[numeric_cols] = self.scaler.inverse_transform(X_imputed[numeric_cols])
        
        return X_imputed
    
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
