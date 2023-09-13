import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import statistics
import seaborn as sns
import tkinter as tk

data = pd.read_csv(r"C:\\Users\manas\Desktop\Manas\Coding\Projects\Misc\Earthquakes-1990-2023.csv")
data = data.drop(columns=["time", "date", "status", "latitude", "longitude", "depth", "tsunami", "place", "data_type"])
data = data.rename(columns={"magnitudo": "magnitude"})
data = data.rename(columns={"significance": "deaths"})
data = data[['state', 'magnitude', 'deaths']]

data_dict_count = {}
for i in range(len(data)):
    state = data.loc[i, 'state']
    magnitude = data.loc[i, 'magnitude']
    if state not in data_dict_count:
        data_dict_count[state] = []
    data_dict_count[state].append(magnitude)

data_dict_deaths = {}
for i in range(len(data)):
    state = data.loc[i, 'state']
    deaths = data.loc[i, 'deaths']
    if state not in data_dict_deaths:
        data_dict_deaths[state] = []
    data_dict_deaths[state].append(deaths)

count = []
states = []
deaths = []

for key, value in data_dict_deaths.items():
    deaths.append(sum(value))

for key, value in data_dict_count.items():
    count.append(len(value))
    states.append(key)

states_count_dict = dict(zip(states, count))
sorted_dict_by_count = dict(sorted(states_count_dict.items(), key=lambda item: item[1], reverse=True))

states_death_dict = dict(zip(states, deaths))
sorted_dict_by_deaths = dict(sorted(states_death_dict.items(), key=lambda item: item[1], reverse=True))

min_magnitudes = []
mean_magnitudes = []
max_magnitudes = []
for key, value in data_dict_count.items():
    min_magnitudes.append(min(value))
    mean_magnitudes.append(round(statistics.mean(value), 2))
    max_magnitudes.append(max(value))

top_4 = {}
counter = 0
for key, value in sorted_dict_by_count.items():
    if counter < 4:
        top_4[key] = value
        counter += 1
    else:
        break

top_4_deaths = {}
counter_deaths = 0
for key, value in sorted_dict_by_deaths.items():
    if counter_deaths < 4:
        top_4_deaths[key] = value
        counter_deaths += 1
    else:
        break

last_10_keys_deaths = list(sorted_dict_by_deaths.keys())[-10:]
sum_deaths = 0
for key in last_10_keys_deaths:
    sum_deaths += sorted_dict_by_deaths[key]
last_10_keys_count = list(sorted_dict_by_count.keys())[-10:]
sum_count = 0
for key in last_10_keys_count:
    sum_count += sorted_dict_by_count[key]

top_4_state = []
top_4_count = []
top_4_deaths_final = []

for key, value in top_4.items():
    top_4_state.append(key)
    top_4_count.append(value)

for value in top_4_deaths.values():
    top_4_deaths_final.append(value)
top_4_deaths_final.append(sum_deaths)
top_4_count.append(sum_count)
top_4_state.append('Other')

# Bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_4_state, top_4_count, color='blue')
plt.xlabel('States')
plt.ylabel('Frequency ')
plt.title('Frequency of Earthquakes (1990)')
plt.tight_layout()
plt.show()

# Pie chart
plt.pie(top_4_deaths_final, labels=top_4_state, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Earthquake-Related Deaths by State (1990)')
plt.show()

# Scatterplot
plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
sns.scatterplot(x=states * 3, y=min_magnitudes + mean_magnitudes + max_magnitudes, hue=["Min"] * len(states) + ["Mean"] * len(states) + ["Max"] * len(states))
plt.xlabel('State')
plt.ylabel('Magnitude')
plt.title('Scatter Plot: Earthquake Magnitude by State in 1990')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Machine Learning
# Create a function to handle the button click event
def generate_sentence():
    input_text = entry.get()
    X = data.drop(columns=['state', 'deaths'])
    y = data['state']
    model = DecisionTreeClassifier()
    model.fit(X.values, y)
    new_data = [[input_text]]
    predictions = model.predict(new_data)
    sentence_label.config(text=f"This kind of magnitude most likely occured in {predictions}")

# Create the main application window
root = tk.Tk()
root.title("Rickter Scale ML")
root.geometry("600x300")
instruction_label = tk.Label(root, text="Enter a value:")
instruction_label.pack()
entry = tk.Entry(root)
entry.pack()
submit_button = tk.Button(root, text="Submit", command=generate_sentence)
submit_button.pack()
sentence_label = tk.Label(root, text="")
sentence_label.pack()
root.mainloop()