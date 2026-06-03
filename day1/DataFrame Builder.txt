import pandas as pd

# Create DataFrame manually
df = pd.DataFrame({
    'Name': ['Rahul', 'Priya', 'Amit', 'Sneha', 'Kiran'],
    'Age': [20, 21, 19, 22, 20],
    'City': ['Chennai', 'Hyderabad', 'Bangalore', 'Mumbai', 'Delhi'],
    'Marks': [85, 45, 72, 38, 90]
})

# Add Result column
df['Result'] = df['Marks'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

# Display DataFrame information
print("DataFrame Head:")
print(df.head())

print("\nShape of DataFrame:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)