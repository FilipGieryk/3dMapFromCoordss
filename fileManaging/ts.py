import pandas as pd
import numpy as np

def csv_relative_to_first(input_csv, output_csv=None, decimals=2):
    # Load CSV
    df = pd.read_csv(input_csv)
    # Get the first data row as reference
    ref = df.iloc[0][['x', 'y', 'z']].values.astype(float)
    # Subtract reference from all rows and round
    df[['x', 'y', 'z']] = np.round(df[['x', 'y', 'z']].astype(float) - ref, decimals)
    # Save or return
    if output_csv:
        df.to_csv(output_csv, index=False, float_format=f'%.{decimals}f')
        print(f"Saved relative CSV to {output_csv}")
    return df

# Example usage:
csv_relative_to_first('output.csv', 'relative_output.csv', decimals=2)