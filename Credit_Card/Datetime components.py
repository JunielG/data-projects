# Assuming df['transaction_date'] is already in datetime format
# If not, first convert it: df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Extract individual datetime components
df['year'] = df['transaction_date'].dt.year
df['month'] = df['transaction_date'].dt.month
df['day'] = df['transaction_date'].dt.day
df['hour'] = df['transaction_date'].dt.hour
df['minute'] = df['transaction_date'].dt.minute
df['second'] = df['transaction_date'].dt.second


# Additional useful datetime features
df['dayofweek'] = df['transaction_date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['quarter'] = df['transaction_date'].dt.quarter
df['is_month_start'] = df['transaction_date'].dt.is_month_start
df['is_month_end'] = df['transaction_date'].dt.is_month_end


# Convert column to datetime if it's not already
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

# Then format it to your desired string format
df['transaction_date'] = df['transaction_date'].dt.strftime('%Y-%m-%d %H:%M:%S')