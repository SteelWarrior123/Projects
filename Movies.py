import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from collections import Counter

mega_data = pd.read_csv(r"C:\Users\manas\Desktop\Manas\Coding\Projects\Movies\imdb_top_1000.csv")
Rename_data = mega_data.drop(columns=["Poster_Link", "Certificate", "Overview", "No_of_Votes", "Meta_score", "Star3", "Star4"])
FINAL = Rename_data.rename(columns={"Series_Title": "Title", "Released_Year":"Year"})
title = list(FINAL['Title'])
year = list(FINAL['Year'])
runtime = list(FINAL['Runtime'])
genre = list(FINAL['Genre'])
rating = list(FINAL['IMDB_Rating'])
director = list(FINAL['Director'])
actor = list(FINAL['Star1'])
gross = list(FINAL['Gross'])

# print(FINAL.columns.tolist())
y_t = {}
year_id = []
title_id = []
for i in range(len(title)):
    year_id = year[i]
    title_id = title[i]
    if year_id in y_t:
        y_t[year_id].append(title_id)
    else:
        y_t[year_id] = [title_id]
year.sort()
y_t = {k: y_t[k] for k in year}
ordered_years = []
ordered_titles_count = []
for keys, values in y_t.items():
     y_t[keys] = len(values)
for keys, values in y_t.items():
     ordered_titles_count.append(values)
     ordered_years.append(keys)

# plt.figure(figsize=(10, 6))
# plt.plot(ordered_years, ordered_titles_count, marker='o', linestyle='-')
# plt.title('When were the hits made?')
# plt.xlabel('Years')
# plt.ylabel('Count')
# plt.xticks(ordered_years[::5], rotation=45)
# plt.tight_layout()
# plt.grid(axis='y', alpha=0.75)
# plt.show()

director_count = []
for i in range(len(director)):
    director_count.append([director[i], director.count(director[i])])
actor_count = []
for i in range(len(actor)):
    actor_count.append([actor[i], actor.count(actor[i])])
def sorter(file_count):
    unique_data = list(set(tuple(sublist) for sublist in file_count))
    unique_data = [list(sublist) for sublist in unique_data]
    sorted_data = sorted(unique_data, key=lambda x: x[1])
    return(sorted_data[-5:])

top5_Actors = sorter(actor_count)
top5_Directors = sorter(director_count)

# def hor_bar(listy, listy1, text):
#     name = []
#     counter = []
#     for item in listy:
#         name.append(item[0])
#         counter.append(item[1])
#     name1 = []
#     counter1 = []
#     for item in listy1:
#         name1.append(item[0])
#         counter1.append(item[1])
#
#     plt.barh(name, counter, color='skyblue', label='Top 5 Actors')
#     plt.barh(name1, counter1, color='orange', label='Top 5 Directors')
#     plt.xlabel('Count')
#     plt.ylabel('Name')
#     plt.title(text)
#     plt.legend()
#     plt.show()
#
# hor_bar(top5_Actors, top5_Directors, "How often were they in/or directed a hit?")

gross_movie = []
for i in range(len(gross)):
    gross_value = int(gross[i].replace(',', ''))
    gross_movie.append([title[i], gross_value])
sorted_gross_movie = sorted(gross_movie, key=lambda x: x[1], reverse=True)
# print(sorted_gross_movie)
#
plt.figure(figsize=(10, 6))
for i in sorted_gross_movie[:10]:
    plt.bar(i[0], i[1], color='blue')
plt.xlabel('Movies')
plt.ylabel('Gross Revenue (Dollars)')
plt.title('Gross Revenue of the top 10 Movies')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# plt.show()

# rating_movie = []
# for i in range(len(gross)):
#     rating_movie.append([title[i], rating[i]])
# sorted_rating_movie = sorted(rating_movie, key=lambda x: x[1], reverse=True)
#
# top_rated_titles = [item[0] for item in sorted_rating_movie[:100]]
# top_rated_runtimes = [int(runtime[title.index(item)].split()[0]) for item in top_rated_titles]
# ratings = [item[1] for item in sorted_rating_movie[:100]]
#
# # Scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(top_rated_runtimes, ratings, color='green')
#
# plt.title('Scatter Plot of IMDB Ratings vs Runtime of the top 100 hits')
# plt.xlabel('Runtime (minutes)')
# plt.ylabel('IMDB Rating')
#
# # Perform linear regression
# slope, intercept, r_value, p_value, std_err = linregress(top_rated_runtimes, ratings)
# line = slope * np.array(top_rated_runtimes) + intercept
# plt.plot(top_rated_runtimes, line, color='red', label=f'Linear Regression (r={r_value:.2f})')
# plt.legend()
#
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()


top_grossing_titles = [item[0] for item in sorted_gross_movie[:100]]
top_grossing_runtimes = [int(runtime[title.index(item)].split()[0]) for item in top_grossing_titles]
gross_values = [item[1] for item in sorted_gross_movie[:100]]

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(top_grossing_runtimes, gross_values, color='blue')

plt.title('Scatter Plot of Gross Revenue vs Runtime for the top 100 hits')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Gross Revenue (Dollars)')

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(top_grossing_runtimes, gross_values)
line = slope * np.array(top_grossing_runtimes) + intercept
plt.plot(top_grossing_runtimes, line, color='red', label=f'Linear Regression (R={r_value:.2f})')

plt.legend()

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Extracting data for popular genres
all_genres = [genre for genres in genre for genre in genres.split(',')]
genre_counts = Counter(all_genres)
top_genres = genre_counts.most_common(8)

# Data for pie chart
# labels, sizes = zip(*top_genres)
# plt.figure(figsize=(8, 8))
# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
# plt.title('Most Popular Genres')
# plt.show()