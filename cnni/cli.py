import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from .imputer import CNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from various file formats."""
    file_path = Path(file_path)
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def identify_context_columns(df: pd.DataFrame) -> tuple:
    """Identify temporal, spatial, and categorical columns."""
    temporal_cols = []
    spatial_cols = []
    categorical_cols = []
    
    print("\nAnalyzing columns:")
    for col in df.columns:
        print(f"Processing column: {col} (type: {df[col].dtype})")
        
        # Check for date columns
        if col.lower() in ['date', 'timestamp', 'time']:
            temporal_cols.append(col)
            print(f"  -> Identified as temporal column")
        
        # Check for spatial columns
        elif col.lower() in ['latitude', 'longitude', 'lat', 'lon', 'lattitude', 'longtitude']:
            spatial_cols.append(col)
            print(f"  -> Identified as spatial column")
        
        # Check for categorical columns
        elif df[col].dtype == 'object' or df[col].dtype == 'category':
            if len(df[col].unique()) < len(df) * 0.5:  # Only if cardinality is reasonable
                categorical_cols.append(col)
                print(f"  -> Identified as categorical column")
    
    return temporal_cols, spatial_cols, categorical_cols

def prepare_context_info(df, temporal_cols, spatial_cols, categorical_cols):
    """Prepare context information from different types of columns."""
    temporal_info = None
    spatial_info = None
    categorical_info = None
    
    print("\nPreparing context information:")
    
    # Handle temporal information
    if temporal_cols:
        print(f"Processing temporal column: {temporal_cols[0]}")
        try:
            # Try with dayfirst=True for DD/MM/YYYY format
            temporal_info = pd.to_numeric(pd.to_datetime(df[temporal_cols[0]], dayfirst=True))
            print("  -> Successfully parsed dates with day-first format")
        except ValueError as e:
            print(f"  -> Warning: Could not parse dates. Error: {e}")
            temporal_info = None
    
    # Handle spatial information
    if spatial_cols and len(spatial_cols) >= 2:
        print(f"Processing spatial columns: {spatial_cols[0]}, {spatial_cols[1]}")
        try:
            # Normalize spatial coordinates
            lat = pd.to_numeric(df[spatial_cols[0]], errors='coerce')
            lon = pd.to_numeric(df[spatial_cols[1]], errors='coerce')
            
            if not (lat.isna().all() or lon.isna().all()):
                # Only standardize if we have valid numbers
                lat_std = lat.std()
                lon_std = lon.std()
                
                if lat_std > 0 and lon_std > 0:
                    spatial_info = np.column_stack([
                        (lat - lat.mean()) / lat_std,
                        (lon - lon.mean()) / lon_std
                    ])
                    print("  -> Successfully processed spatial coordinates")
                else:
                    print("  -> Warning: Zero standard deviation in coordinates")
            else:
                print("  -> Warning: All coordinates are NaN")
        except Exception as e:
            print(f"  -> Error processing spatial data: {e}")
    
    # Handle categorical information
    if categorical_cols:
        print(f"Processing categorical columns: {', '.join(categorical_cols)}")
        categorical_data = []
        for col in categorical_cols:
            try:
                # Get unique values count
                unique_count = df[col].nunique()
                if unique_count > 100:  # Skip if too many categories
                    print(f"  -> Skipping {col}: too many unique values ({unique_count})")
                    continue
                    
                encoded = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                if not encoded.empty:
                    categorical_data.append(encoded)
                    print(f"  -> Successfully encoded {col}")
            except Exception as e:
                print(f"  -> Warning: Could not encode column {col}. Error: {e}")
        
        if categorical_data:
            try:
                categorical_info = np.column_stack([data.values for data in categorical_data])
                print(f"  -> Successfully combined {len(categorical_data)} categorical columns")
            except Exception as e:
                print(f"  -> Error combining categorical data: {e}")
    
    return temporal_info, spatial_info, categorical_info

def save_plots(df_original, df_imputed, output_dir='plots'):
    """Save visualization plots for imputation results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_theme()
    
    # Get numeric columns with missing values
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    missing_cols = [col for col in numeric_cols if df_original[col].isna().any()]
    
    print("\nGenerating plots...")
    
    # 1. Distribution plots for each imputed column
    for col in missing_cols:
        plt.figure(figsize=(12, 6))
        
        # Plot original non-missing values
        sns.kdeplot(
            data=df_original[~df_original[col].isna()][col],
            label='Original (non-missing)',
            color='blue',
            alpha=0.5
        )
        
        # Plot imputed values
        imputed_mask = df_original[col].isna()
        if imputed_mask.any():
            sns.kdeplot(
                data=df_imputed[imputed_mask][col],
                label='Imputed values',
                color='red',
                alpha=0.5
            )
        
        plt.title(f'Distribution Comparison for {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Missing value patterns
    plt.figure(figsize=(12, 6))
    missing = df_original[missing_cols].isna().sum().sort_values(ascending=True)
    sns.barplot(x=missing.values, y=missing.index)
    plt.title('Missing Values by Column')
    plt.xlabel('Number of Missing Values')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Spatial distribution of missing values (if latitude/longitude available)
    if 'Lattitude' in df_original.columns and 'Longtitude' in df_original.columns:
        for col in missing_cols:
            plt.figure(figsize=(12, 8))
            
            # Plot non-missing values
            plt.scatter(
                df_original[~df_original[col].isna()]['Longtitude'],
                df_original[~df_original[col].isna()]['Lattitude'],
                alpha=0.5,
                label='Original',
                color='blue',
                s=20
            )
            
            # Plot locations of imputed values
            plt.scatter(
                df_original[df_original[col].isna()]['Longtitude'],
                df_original[df_original[col].isna()]['Lattitude'],
                alpha=0.5,
                label='Imputed',
                color='red',
                s=20
            )
            
            plt.title(f'Spatial Distribution of Missing Values - {col}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'spatial_distribution_{col}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Correlation heatmap of imputed dataset
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df_imputed[numeric_cols].corr(),
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        square=True
    )
    plt.title('Correlation Matrix After Imputation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved in {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='CNNI - Contextual Nearest Neighbor Imputation')
    parser.add_argument('input_file', help='Path to input data file (CSV or Excel)')
    parser.add_argument('--max-k', type=int, default=10, help='Maximum number of neighbors')
    parser.add_argument('--temporal-weight', type=float, default=0.3, help='Weight for temporal context')
    parser.add_argument('--spatial-weight', type=float, default=0.3, help='Weight for spatial context')
    parser.add_argument('--categorical-weight', type=float, default=0.4, help='Weight for categorical context')
    parser.add_argument('--output', help='Path to save imputed data (optional)')
    parser.add_argument('--plot-dir', default='plots', help='Directory to save plots (default: plots)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading data from {args.input_file}...")
    try:
        df = pd.read_csv(args.input_file)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Store original data for comparison
    df_original = df.copy()
    
    # Print missing value statistics
    print("\nMissing value statistics:")
    total_missing = df.isna().sum().sum()
    total_cells = df.size
    print(f"Total missing values: {total_missing}")
    print(f"Missing rate: {total_missing/total_cells:.2%}")
    
    print("\nMissing values per column:")
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_count/len(df):.2%})")
    
    # Identify context columns
    print("\nIdentifying context columns...")
    temporal_cols, spatial_cols, categorical_cols = identify_context_columns(df)
    
    if temporal_cols:
        print(f"  Temporal columns: {', '.join(temporal_cols)}")
    if spatial_cols:
        print(f"  Spatial columns: {', '.join(spatial_cols)}")
    if categorical_cols:
        print(f"  Categorical columns: {', '.join(categorical_cols)}")
    
    # Prepare context information
    temporal_info, spatial_info, categorical_info = prepare_context_info(
        df, temporal_cols, spatial_cols, categorical_cols
    )
    
    # Get numeric columns for imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nColumns to be imputed: {', '.join(numeric_cols)}")
    
    # Initialize imputer with more conservative parameters
    print("\nInitializing CNNI model...")
    imputer = CNNImputer(
        max_k=min(args.max_k, len(df) // 10),  # Limit k based on dataset size
        temporal_weight=args.temporal_weight,
        spatial_weight=args.spatial_weight,
        categorical_weight=args.categorical_weight
    )
    
    # Perform imputation
    print("\nPerforming imputation...")
    try:
        df_imputed = imputer.fit_transform(
            df[numeric_cols],
            temporal_info=temporal_info,
            spatial_info=spatial_info,
            categorical_info=categorical_info
        )
        print("Imputation completed successfully")
    except Exception as e:
        print(f"Error during imputation: {e}")
        return
    
    # Save imputed data
    if args.output:
        print(f"\nSaving imputed data to {args.output}...")
        try:
            # Combine imputed numeric columns with non-numeric columns
            df_final = df.copy()
            df_final[numeric_cols] = df_imputed
            df_final.to_csv(args.output, index=False)
            print("Successfully saved imputed data")
            
            # Print summary of changes
            print("\nSummary of changes:")
            for col in numeric_cols:
                original_missing = df[col].isna().sum()
                if original_missing > 0:
                    print(f"  {col}: Imputed {original_missing} values")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    # Generate and save plots
    save_plots(df_original, df_final, args.plot_dir)

if __name__ == "__main__":
    main()
