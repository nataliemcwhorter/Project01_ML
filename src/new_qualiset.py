import pandas as pd

# Read the CSV files
qualifying_df = pd.read_csv('../data/qualifying.csv')
races_df = pd.read_csv('../data/races.csv')

# Select only the columns we need from races
races_subset = races_df[['raceId', 'year', 'date', 'circuitId']]

# Merge the datasets on raceId
merged_df = qualifying_df.merge(races_subset, on='raceId', how='left')

# Reorder columns to have the new columns after the existing qualifying columns
# Original qualifying columns: qualifyId, raceId, driverId, constructorId, number, position, q1, q2, q3
# New columns: year, date, circuitId
column_order = ['qualifyId', 'circuitId', 'raceId', 'driverId', 'constructorId', 'year', 'date','number',
                'position', 'q1', 'q2', 'q3']

merged_df = merged_df[column_order]

# Save the merged dataset
merged_df.to_csv('../data/qualifying_enhanced.csv', index=False)

print("Dataset merged successfully!")
print(f"Original qualifying dataset shape: {qualifying_df.shape}")
print(f"Merged dataset shape: {merged_df.shape}")
print("\nFirst few rows of the merged dataset:")
print(merged_df.head())

# Check for any missing values in the new columns
print(f"\nMissing values in new columns:")
print(f"Year: {merged_df['year'].isna().sum()}")
print(f"Date: {merged_df['date'].isna().sum()}")
print(f"CircuitId: {merged_df['circuitId'].isna().sum()}")
