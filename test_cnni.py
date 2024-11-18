import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from cnni import CNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=1000, n_features=5, missing_rate=0.2):
    """Generate synthetic data with missing values."""
    # Generate regression dataset
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )
    
    # Convert to DataFrame
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    
    # Add temporal information
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Add spatial information
    df['latitude'] = np.random.uniform(low=30, high=50, size=n_samples)
    df['longitude'] = np.random.uniform(low=-120, high=-70, size=n_samples)
    
    # Add categorical information
    df['category'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    
    # Create missing values
    mask = np.random.random(size=df.shape) < missing_rate
    df_missing = df.copy()
    df_missing[mask] = np.nan
    
    return df, df_missing, mask

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    df_true, df_missing, mask = generate_synthetic_data()
    
    # Initialize and fit CNNI
    print("\nInitializing CNNI model...")
    imputer = CNNImputer(
        max_k=10,
        temporal_weight=0.3,
        spatial_weight=0.3,
        categorical_weight=0.4
    )
    
    # Prepare context information
    temporal_info = pd.to_numeric(df_missing['timestamp'])
    spatial_info = df_missing[['latitude', 'longitude']]
    categorical_info = pd.get_dummies(df_missing['category'])
    
    # Perform imputation
    print("Performing imputation...")
    numeric_cols = [col for col in df_missing.columns if col.startswith('feature_')]
    df_imputed = imputer.fit_transform(
        df_missing[numeric_cols],
        temporal_info=temporal_info,
        spatial_info=spatial_info,
        categorical_info=categorical_info
    )
    
    # Evaluate performance
    print("\nEvaluating performance...")
    rmse, mae = imputer.evaluate(
        df_true[numeric_cols],
        df_imputed,
        mask[:, :len(numeric_cols)]
    )
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Plot results
    print("\nPlotting results...")
    imputer.plot_results(
        df_true[numeric_cols],
        df_imputed,
        mask[:, :len(numeric_cols)]
    )
    
    # Additional visualization: Distribution comparison
    plt.figure(figsize=(15, 5))
    
    # Plot original vs imputed distributions for first feature
    feature = numeric_cols[0]
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df_true[feature], label='Original', alpha=0.5)
    sns.kdeplot(data=df_imputed[feature], label='Imputed', alpha=0.5)
    plt.title(f'Distribution Comparison - {feature}')
    plt.legend()
    
    # Plot imputation error distribution
    plt.subplot(1, 2, 2)
    errors = df_true[numeric_cols] - df_imputed
    sns.boxplot(data=errors)
    plt.title('Imputation Error Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
