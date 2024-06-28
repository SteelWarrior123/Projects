import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

data_df = pd.read_csv(r"C:\Users\manas\Desktop\Manas\Coding\Projects\Heart Attack\Heart_Disease_Prediction.csv")
pd.set_option('display.max_columns', None)
map = {
    "Presence": 1,
    "Absence": 0,
}
data_df["Heart Disease"] = data_df["Heart Disease"].map(map)
male_pres = data_df.query('`Heart Disease` == 1 and Sex == 1')
female_pres = data_df.query('`Heart Disease` == 1 and Sex == 0')

# Pie chart for overall survival distribution
label = ['Male', 'Female']
color = ['lightblue', 'lightpink']
plt.figure(figsize=(6, 6))
plt.pie([len(male_pres), len(female_pres)], labels=label, wedgeprops={'edgecolor': 'black', 'linewidth': 1}, colors=color, autopct='%1.1f%%', startangle=90, explode=(0.1, 0), shadow=True)
plt.axis('equal')
plt.title('Presence of Heart Disease by gender')


pres = data_df[data_df['Heart Disease'] == 1]
abs = data_df[data_df['Heart Disease'] == 0]

# Prepare the data
grouped_abs = abs.groupby('Chest pain type')['Heart Disease'].count()
grouped_pres = pres.groupby('Chest pain type')['Heart Disease'].count()

chest_pain_labels = {
    1: 'Typical angina',
    2: 'Atypical angina',
    3: 'Non-anginal pain',
    4: 'Asymptomatic'
}

grouped_abs.index = grouped_abs.index.map(chest_pain_labels)
grouped_pres.index = grouped_pres.index.map(chest_pain_labels)

plt.figure(figsize=(10, 6))
bar_width = 0.4
r1 = np.arange(len(grouped_abs))
r2 = [x + bar_width for x in r1]
plt.bar(r1, grouped_abs, color='lightblue', width=bar_width, edgecolor='black', label='Absence of Heart Disease')
plt.bar(r2, grouped_pres, color='coral', width=bar_width, edgecolor='black', label='Presence of Heart Disease')
plt.xlabel('Chest pain type')
plt.ylabel('Count')
plt.title('Heart Disease by Chest pain type')
plt.xticks([r + bar_width / 2 for r in range(len(grouped_abs))], grouped_abs.index)
plt.legend()


# Create a figure and axes
fig, ax = plt.subplots()
colors = ['lightcoral', 'lightgreen']
# Plot boxplot
bplot = ax.boxplot([pres['BP'], abs['BP']], patch_artist=True)  
# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
for median in bplot['medians']:
    median.set_color('black')
ax.set_xlabel('Variables')
ax.set_ylabel('Blood Pressure')
ax.set_title('Blood Pressure of patients')
ax.set_xticklabels(['Presence of Heart Disease', 'Absence of Heart Disease'])
ax.grid(True)

# Create a figure and axes
fig, ax = plt.subplots()
bplot1 = ax.boxplot([pres['Max HR'], abs['Max HR']], patch_artist=True)  
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
for median in bplot1['medians']:
    median.set_color('black')
ax.set_xlabel('Variables')
ax.set_ylabel('Heart Rate')
ax.set_title('Max Heart Rate of patients')
ax.set_xticklabels(['Presence of Heart Disease', 'Absence of Heart Disease'])
ax.grid(True)

grouped_abs1 = abs.groupby('FBS over 120')['Heart Disease'].count()
grouped_pres1 = pres.groupby('FBS over 120')['Heart Disease'].count()

# Mapping FBS labels
FBS_labels = {0: 'Below 120', 1: 'Above 120'}
grouped_abs1.index = grouped_abs1.index.map(FBS_labels)
grouped_pres1.index = grouped_pres1.index.map(FBS_labels)


plt.figure(figsize=(10, 6))
bar_width = 0.4
r1 = np.arange(len(grouped_abs1))
r2 = [x + bar_width for x in r1]
plt.bar(r1, grouped_abs1, color='lightblue', width=bar_width, edgecolor='black', label='Absence of Heart Disease')
plt.bar(r2, grouped_pres1, color='coral', width=bar_width, edgecolor='black', label='Presence of Heart Disease')
plt.xlabel('Fasting Blood Sugar')
plt.ylabel('Count')
plt.title('Heart Disease by Fasting Blood Sugar')
plt.xticks([r + bar_width / 2 for r in range(len(grouped_abs1))], grouped_abs1.index)
plt.legend()

# Prepare the data
X = data_df.drop('Heart Disease', axis=1)
y = data_df['Heart Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
sorted_feature_names = [feature_names[i] for i in indices]

# Plot the feature importances
plt.figure(figsize=(12, 8))
plt.barh(range(X.shape[1]), importances[indices], color='lightblue', edgecolor='black')
plt.yticks(range(X.shape[1]), sorted_feature_names)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Predicting Heart Disease')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()
