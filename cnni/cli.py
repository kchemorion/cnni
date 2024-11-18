import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from .imputer import CNNImputer

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
    
    for col in df.columns:
        # Check for temporal columns
        if df[col].dtype == 'datetime64[ns]' or 'time' in col.lower() or 'date' in col.lower():
            temporal_cols.append(col)
        # Check for spatial columns
        elif 'lat' in col.lower() or 'lon' in col.lower() or 'location' in col.lower():
            spatial_cols.append(col)
        # Check for categorical columns
        elif df[col].dtype == 'object' or df[col].dtype == 'category':
            categorical_cols.append(col)
    
    return temporal_cols, spatial_cols, categorical_cols

def prepare_context_info(
    df: pd.DataFrame,
    temporal_cols: list,
    spatial_cols: list,
    categorical_cols: list
) -> tuple:
    """Prepare context information for CNNI."""
    temporal_info = None
    spatial_info = None
    categorical_info = None
    
    # Prepare temporal information
    if temporal_cols:
        temporal_info = pd.to_numeric(pd.to_datetime(df[temporal_cols[0]]))
    
    # Prepare spatial information
    if len(spatial_cols) >= 2:
        spatial_info = df[spatial_cols[:2]]  # Use first two spatial columns (assumed lat/lon)
    
    # Prepare categorical information
    if categorical_cols:
        categorical_info = pd.get_dummies(df[categorical_cols])
    
    return temporal_info, spatial_info, categorical_info

def main():
    parser = argparse.ArgumentParser(description='CNNI - Contextual Nearest Neighbor Imputation')
    parser.add_argument('input_file', help='Path to input data file (CSV or Excel)')
    parser.add_argument('--max-k', type=int, default=10, help='Maximum number of neighbors')
    parser.add_argument('--temporal-weight', type=float, default=0.3, help='Weight for temporal context')
    parser.add_argument('--spatial-weight', type=float, default=0.3, help='Weight for spatial context')
    parser.add_argument('--categorical-weight', type=float, default=0.4, help='Weight for categorical context')
    parser.add_argument('--output', help='Path to save imputed data (optional)')
    parser.add_argument('--plot', action='store_true', help='Show imputation results plots')
    parser.add_argument('--test-mode', action='store_true', help='Test mode: artificially create missing values')
    parser.add_argument('--missing-rate', type=float, default=0.2, help='Missing rate for test mode (default: 0.2)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    df = load_data(args.input_file)
    
    # Store original data and create mask
    if args.test_mode:
        print(f"Test mode: Creating artificial missing values (rate: {args.missing_rate})...")
        df_true = df.copy()
        mask = np.random.random(size=df.shape) < args.missing_rate
        df_missing = df.copy()
        df_missing.mask(mask, inplace=True)
    else:
        # For real missing values
        df_missing = df.copy()
        mask = df_missing.isna()
        df_true = df_missing.copy()  # We don't have true values in this case
    
    # Print missing value statistics
    total_missing = mask.sum().sum()
    total_cells = mask.size
    print(f"\nMissing value statistics:")
    print(f"Total missing values: {total_missing}")
    print(f"Missing rate: {total_missing/total_cells:.2%}")
    print("\nMissing values per column:")
    for col in df_missing.columns:
        missing_count = mask[col].sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_count/len(df_missing):.2%})")
    
    # Identify context columns
    print("\nIdentifying context columns...")
    temporal_cols, spatial_cols, categorical_cols = identify_context_columns(df_missing)
    
    if temporal_cols:
        print(f"  Temporal columns: {', '.join(temporal_cols)}")
    if spatial_cols:
        print(f"  Spatial columns: {', '.join(spatial_cols)}")
    if categorical_cols:
        print(f"  Categorical columns: {', '.join(categorical_cols)}")
    
    # Prepare context information
    temporal_info, spatial_info, categorical_info = prepare_context_info(
        df_missing, temporal_cols, spatial_cols, categorical_cols
    )
    
    # Initialize imputer
    print("\nInitializing CNNI model...")
    imputer = CNNImputer(
        max_k=args.max_k,
        temporal_weight=args.temporal_weight,
        spatial_weight=args.spatial_weight,
        categorical_weight=args.categorical_weight
    )
    
    # Get numeric columns for imputation
    numeric_cols = df_missing.select_dtypes(include=[np.number]).columns
    print(f"\nColumns to be imputed: {', '.join(numeric_cols)}")
    
    # Perform imputation
    print("\nPerforming imputation...")
    df_imputed = imputer.fit_transform(
        df_missing[numeric_cols],
        temporal_info=temporal_info,
        spatial_info=spatial_info,
        categorical_info=categorical_info
    )
    
    # Evaluate performance if in test mode
    if args.test_mode:
        print("\nEvaluating performance...")
        rmse, mae = imputer.evaluate(
            df_true[numeric_cols],
            df_imputed,
            mask[:, df_missing.columns.get_indexer(numeric_cols)]
        )
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    # Plot results if requested
    if args.plot:
        print("\nPlotting results...")
        if args.test_mode:
            imputer.plot_results(
                df_true[numeric_cols],
                df_imputed,
                mask[:, df_missing.columns.get_indexer(numeric_cols)]
            )
        else:
            print("Plotting not available in normal mode (no true values for comparison)")
    
    # Save imputed data if output path is provided
    if args.output:
        print(f"\nSaving imputed data to {args.output}...")
        # Combine imputed numeric columns with non-numeric columns
        df_final = df_missing.copy()
        df_final[numeric_cols] = df_imputed
        df_final.to_csv(args.output, index=False)
        print("Done!")
        
        # Print summary of changes
        print("\nSummary of changes:")
        for col in numeric_cols:
            original_missing = df_missing[col].isna().sum()
            final_missing = df_final[col].isna().sum()
            if original_missing > 0:
                print(f"  {col}: Imputed {original_missing} values")
                if final_missing > 0:
                    print(f"    {final_missing} values still missing")

if __name__ == "__main__":
    main()
