import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
import statistics
from collections import OrderedDict
from collections import Counter
import chart_studio.plotly as py
import chart_studio.tools as tls

mega_data = pd.read_csv(r"C:\Users\manas\Desktop\Manas\Coding\Projects\Crime in Toronto\Homicides_Open_Data_ASR_RC_TBL_002.csv")
data = mega_data.drop(columns=["X", "Y", "OBJECTID", "EVENT_UNIQUE_ID", "OCC_DATE", "OCC_DAY", "DIVISION", "NEIGHBOURHOOD_140", "HOOD_140"])
data = data.rename(columns={"OCC_YEAR": "Year", "OCC_MONTH":"Month", "OCC_DOW":"DOW", "OCC_DOY":"DOY", "HOOD_158":"HOODID", "NEIGHBOURHOOD_158":"Neighbourhood", "LONG_WGS84":"LONG", "LAT_WGS84":"LAT",  "HOMICIDE_TYPE":"Type"})

long = data['LONG']
lat = data['LAT']
ID = data['HOODID']
name = data['Neighbourhood']
type = data['Type']
month = data['Month']

hoods_ll = data[['HOODID', 'LAT', 'LONG']]
HOODS_LONG = {}
HOODS_LAT = {}

for i in range(len(hoods_ll)):
    hood_id = ID[i]
    current_long = long[i]
    if hood_id in HOODS_LONG:
        HOODS_LONG[hood_id].append(current_long)
    else:
        HOODS_LONG[hood_id] = [current_long]

for i in range(len(hoods_ll)):
    hood_id = ID[i]
    current_lat = lat[i]
    if hood_id in HOODS_LAT:
        HOODS_LAT[hood_id].append(current_lat)
    else:
        HOODS_LAT[hood_id] = [current_lat]

avg_lat = []
avg_long = []
hood_count = []

for values in HOODS_LONG.values():
    avg_long.append(statistics.mean(values))

for values in HOODS_LAT.values():
    avg_lat.append(statistics.mean(values))

for values in HOODS_LAT.values():
    hood_count.append(len(values))

names = []
for i in name:
    if i in names:
        continue
    names.append(i)

final_data = {'Long': avg_long, 'Lat': avg_lat, 'Deaths': hood_count, 'Neighbourhood': names}
df = pd.DataFrame(final_data)
fig = px.scatter_mapbox(df, lon=df['Long'], lat=df['Lat'], zoom=10, color=df['Deaths'], hover_name="Neighbourhood", width=1200, height=900, title='Crime Data Map in Toronto (2004-2022)')
fig.update_layout(mapbox_style="open-street-map")
fig.update_traces(marker=dict(size=15))
# fig.show()
username = 'Manast'
api_key = 'ruWTdLItjxKp2fg7dX8e'

py.sign_in(username, api_key)
chart_url = py.plot(fig, filename='crime_data_map_toronto', auto_open=False)

print("Chart URL:", chart_url)

#Murders along a week
DOW = data['DOW']
DOW_COUNT = {}

for i in range(len(hoods_ll)):
    current_DOW = DOW[i]
    current_ID = ID[i]
    if current_DOW in DOW_COUNT:
        DOW_COUNT[current_DOW].append(current_ID)
    else:
        DOW_COUNT[current_DOW] = [current_ID]

for keys, values in DOW_COUNT.items():
    DOW_COUNT[keys] = len(values)
DOW_COUNT = {'Monday': 185, 'Tuesday': 155, 'Wednesday': 177,  'Thursday': 142, 'Friday': 204, 'Saturday': 230, 'Sunday': 228}

Days = []
D_Freq = []
for keys, values in DOW_COUNT.items():
    Days.append(keys)
    D_Freq.append(values)

plt.figure(figsize=(10, 6))
plt.bar(Days, D_Freq, color='lightblue', edgecolor='black')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.title('Count of Homocide across the week (2004-2022)')
plt.tight_layout()
plt.grid(axis='y', alpha=0.75)
plt.show()

#Linechart
YEAR = data['Year']
YEAR_COUNT = {}

for i in range(len(hoods_ll)):
    current_YEAR = YEAR[i]
    current_ID = ID[i]
    if current_YEAR in YEAR_COUNT:
        YEAR_COUNT[current_YEAR].append(current_ID)
    else:
        YEAR_COUNT[current_YEAR] = [current_ID]

for keys, values in YEAR_COUNT.items():
    YEAR_COUNT[keys] = len(values)
Years = []
Y_Freq = []
for pair in sorted(YEAR_COUNT.items()):
    Years.append(pair[0])
    Y_Freq.append(pair[1])

plt.figure(figsize=(10, 6))
plt.plot(Years, Y_Freq, marker='o', linestyle='-')
plt.title('Homocide in Toronto (2004-2022)')
plt.xlabel('Years')
plt.ylabel('Count')
plt.xticks(Years, rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.show()

# Murder types
murder_types = {}
for i in range(len(hoods_ll)):
    current_YEAR = YEAR[i]
    current_type = type[i]
    if current_YEAR in murder_types:
        murder_types[current_YEAR].append(current_type)
    else:
        murder_types[current_YEAR] = [current_type]

shooting = []
stabbing = []
other = []
combined_type = []

ordered_data = OrderedDict(sorted(murder_types.items()))
for year, values in ordered_data.items():
    combined_type.append([values.count('Shooting'), values.count('Stabbing'), values.count('Other')])

for i in range(len(combined_type)):
    shooting.append(combined_type[i][0])
    stabbing.append(combined_type[i][1])
    other.append(combined_type[i][2])

fig, ax = plt.subplots(figsize=(10, 6))
bottom_shooting = [0] * len(Years)
bottom_stabbing = [s for s in shooting]
bottom_other = [s + st for s, st in zip(shooting, stabbing)]
ax.bar(Years, shooting, color='blue', edgecolor='black', label='Shooting', bottom=bottom_shooting)
ax.bar(Years, stabbing, color='green', edgecolor='black', label='Stabbing', bottom=bottom_stabbing)
ax.bar(Years, other, color='red', edgecolor='black', label='Other', bottom=bottom_other)
for year, s, st, o in zip(Years, shooting, stabbing, other):
    ax.text(year, s/2, str(s), ha='center', va='center', fontsize=10)
    ax.text(year, s + st/2, str(st), ha='center', va='center', fontsize=10)
    ax.text(year, s + st + o/2, str(o), ha='center', va='center', fontsize=10)
ax.set_xlabel('Years')
ax.set_ylabel('Count')
ax.set_title('Count of Homicides by Type (2004-2022)')
ax.set_xticks(Years)
ax.set_xticklabels(Years, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

#Histogram
month_count = []
monthy = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
for i in monthy:
    month_count.append((month == i).sum())

# Create the histogram
plt.bar(monthy, month_count, color='#e76f51', edgecolor='black')
plt.xlabel('Months')
plt.ylabel('Count')
plt.title('Count of Homocide on average across a year')
plt.xticks(rotation=45)
custom_y_ticks = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
plt.ylim(min(custom_y_ticks), max(custom_y_ticks))
plt.yticks(custom_y_ticks)
plt.grid(axis='y', alpha=0.75)
plt.show()