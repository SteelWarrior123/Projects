import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load the CSV file into a DataFrame
df = pd.read_csv(r"C:/Users/manas/Desktop/Manas/Coding/Projects/World Temp/GlobalLandTemperaturesByCity.csv")

# Rename columns for convenience
df.rename(columns={'dt': 'date', 'AverageTemperature': 'temp'}, inplace=True)

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Fill any NaN values
df = df.bfill()

# Extract year, month, and other features from the index
def create_features(df):
    df = df.copy()
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_features(df)
df = df[df['year'] != 2013]
df = df[df['year'] != 1743]

# Select only the necessary columns for aggregation
df_numeric = df[['year', 'month', 'temp']]

# Remove duplicates by taking the mean temperature for duplicate entries
df_grouped = df_numeric.groupby(['year', 'month']).mean().reset_index()

# Convert 'month' column to integer to ensure proper formatting
df_grouped['month'] = df_grouped['month'].astype(int)

# Create a single continuous sequence of temperatures
df_grouped['date'] = df_grouped.apply(lambda row: f"{int(row['year'])}-{int(row['month']):02}", axis=1)
df_grouped['date'] = pd.to_datetime(df_grouped['date'], format='%Y-%m')

# Sort by date
df_grouped.sort_values(by='date', inplace=True)
df_grouped = df_grouped.loc[df_grouped['year'] > 1750]

train = df_grouped.loc[df_grouped['year'] < 1975]
test = df_grouped.loc[df_grouped['year'] >= 1975]

FEATURES = ['month', 'year']
TARGET = 'temp'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

plt.figure(figsize=(15, 5))
plt.plot(train['date'], train['temp'], label='Train set', linewidth = 1)
plt.plot(test['date'], test['temp'], label='Test set', linewidth = 1)
plt.axvline(1975, color='black', ls='--')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.title('Train/Test set data split')

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

# Plot feature importance
fi = pd.DataFrame(data=reg.feature_importances_,
                  index=FEATURES,
                  columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')

# Predictions
test['prediction'] = reg.predict(X_test)

# # Merge predictions back into df_grouped
df_grouped = df_grouped.merge(test[['prediction', 'date']], how='left', on='date')
# # Plotting
plt.figure(figsize=(15, 5))
plt.plot(test['date'], test['temp'], label='True Data', linewidth = 1)
plt.plot(test['date'], test['prediction'], label='Predictions', linewidth = 1)
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Raw Data and Predictions')
plt.legend(loc='lower right')
plt.show()

score = np.sqrt(mean_squared_error(test['temp'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')





