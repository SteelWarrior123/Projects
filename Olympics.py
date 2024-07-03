import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px


pd.set_option('display.max_columns', None)
df = pd.read_csv(r"C:\Users\manas\Desktop\Manas\Coding\Projects\Olympics\athlete_events.csv")

tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_participating_nations_at_the_Summer_Olympic_Games")
countries_2024_olympics = ["Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", 
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", 
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", 
    "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", 
    "Chile", "China", "Colombia", "Comoros", "Congo", "Cook Islands", "Costa Rica", "Croatia", 
    "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", 
    "East Timor", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", 
    "Eswatini", "Ethiopia", "Federated States of Micronesia", "Fiji", "Finland", "France", "Gabon", 
    "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guam", "Guatemala", "Guinea", 
    "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", 
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", 
    "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", 
    "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", 
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", 
    "Mexico", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", 
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", 
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", 
    "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Puerto Rico", "Qatar", "Republic of the Congo", 
    "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", 
    "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", 
    "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", 
    "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", 
    "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Togo", "Tonga", "Trinidad and Tobago", 
    "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", 
    "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", 
    "Yemen", "Zambia", "Zimbabwe"]

df = df[df['Team'].isin(countries_2024_olympics)]
df = df[(df['Season'] != 'Winter')]

df_medal_dummies = pd.get_dummies(df['Medal'], prefix='Medal').astype(int)
df_sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex').astype(int)
df = pd.concat([df, df_medal_dummies, df_sex_dummies], axis=1)
medal_winners = df[~df['Medal'].isna()]

# Country
by_country = medal_winners.groupby(['Team', 'NOC']).size().reset_index(name='Medal')

top_5_most_ath = (df.groupby(['Team', 'NOC']).size().reset_index(name='Name')).sort_values(by='Name', ascending=False)
top_5_team_medal = by_country.sort_values(by='Medal', ascending=False)
# Athlete
by_athlete = medal_winners.groupby(['Name', 'Team', 'Sex']).size().reset_index(name='Medal')

top_3_ath = by_athlete.sort_values(by = 'Medal', ascending=False)
youngest = df.sort_values(by = 'Age')
oldest = df[df['Age'] == df['Age'].max()]
oldest_winner =  medal_winners[medal_winners['Age'] == medal_winners['Age'].max()]
athlete_participation_count = df.groupby('Name')['Year'].nunique().reset_index(name='Editions').sort_values('Editions').tail(1)

# Sports
sport_count = df.groupby('Sport').size().reset_index(name='Count').sort_values(by= 'Count', ascending=False).head(3)
mean_age = df.groupby('Sport').mean('Age').reset_index()
mean_age = mean_age[['Sport', 'Age']].round(2).sort_values(by= 'Age', ascending=False)
mean_height = df.groupby('Sport').mean('Height').reset_index()
mean_height = mean_height[['Sport', 'Height']].round(2).sort_values(by= 'Height', ascending=False)

# By year
by_year = df.groupby('Year').sum()
by_year_sex = by_year[['Sex_F', 'Sex_M']]

mean_age_medal = medal_winners.groupby(['Year', 'Medal'])['Age'].mean().reset_index()
mean_age_pivot = mean_age_medal.pivot(index='Year', columns='Medal', values='Age')
mean_age_pivot = mean_age_pivot[['Gold', 'Silver', 'Bronze']]

sns.set_theme()
plt.figure(figsize=(12, 6))

# Define the colors for each medal type
medal_colors = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}
for medal in mean_age_pivot.columns:
    sns.lineplot(data=mean_age_pivot, x=mean_age_pivot.index, y=medal, label=medal, color=medal_colors[medal])
plt.ylabel('Mean Age')
plt.xlabel('Year')
plt.legend(title='Medal')
plt.show()

medal_counts = medal_winners.groupby(['Team', 'Year'])['Medal'].count().reset_index(name='MedalCount')
medal_counts_pivot = medal_counts.pivot(index='Team', columns='Year', values='MedalCount').fillna(0)
total_medal_counts = medal_counts_pivot.sum(axis=1).reset_index(name='TotalMedalCount')
top_5_countries = total_medal_counts.nlargest(5, 'TotalMedalCount')['Team']
medal_counts_top_5_pivot = medal_counts_pivot.loc[top_5_countries]
medal_counts_top_5_pivot_transposed = medal_counts_top_5_pivot.T

# By host 
cities = [
    "Amsterdam", "Antwerpen", "Athina", "Atlanta", "Barcelona", "Beijing", 
    "Berlin", "Helsinki", "London", "Los Angeles", "Melbourne", "Mexico City", 
    "Montreal", "Moskva", "Munich", "Paris", "Rio de Janeiro", "Roma", 
    "Seoul", "St. Louis", "Stockholm", "Sydney", "Tokyo"]
city_to_country = {
    "Amsterdam": "Netherlands",
    "Antwerpen": "Belgium",
    "Athina": "Greece",
    "Atlanta": "United States",
    "Barcelona": "Spain",
    "Beijing": "China",
    "Berlin": "Germany",
    "Helsinki": "Finland",
    "London": "United Kingdom",
    "Los Angeles": "United States",
    "Melbourne": "Australia",
    "Mexico City": "Mexico",
    "Montreal": "Canada",
    "Moskva": "Russia",
    "Munich": "Germany",
    "Paris": "France",
    "Rio de Janeiro": "Brazil",
    "Roma": "Italy",
    "Seoul": "South Korea",
    "St. Louis": "United States",
    "Stockholm": "Sweden",
    "Sydney": "Australia",
    "Tokyo": "Japan"
}

# Create a DataFrame from the dictionary for easier manipulation
cities_df = pd.DataFrame(list(city_to_country.items()), columns=['City', 'Country'])
unique_countries = cities_df['Country'].unique()
countries_df = pd.DataFrame({'Country': unique_countries})

# # Plotting with Plotly Express
# fig = px.choropleth(countries_df, locations="Country", locationmode="country names", 
#                     color_discrete_sequence=["lightgreen"], scope="world",
#                     title="Countries that have hosted the Olympics")

# fig.show()

# plt.figure(figsize=(20, 10))
# markers = ['o', 's', 'D', '^', 'v']
# for idx, country in enumerate(medal_counts_top_5_pivot_transposed.columns):
#     sns.lineplot(x=medal_counts_top_5_pivot_transposed.index,
#                  y=medal_counts_top_5_pivot_transposed[country],
#                  marker=markers[idx],
#                  label=country)

# plt.title('Year-wise Medal Count by Top 5 Countries')
# plt.grid()
# plt.xlabel('Year')
# plt.ylabel('Number of Medals')
# plt.legend(title='Country', loc='upper left')
# plt.show()

# sns.set_theme()
# sns.lineplot(x = 'Year', y = 'Sex_M', data = by_year_sex, label='Male', color='lightblue')
# sns.lineplot(x = 'Year', y = 'Sex_F', data = by_year_sex, label='Female', color='lightpink')
# plt.ylabel('Count')
# plt.xlabel('Year')
# plt.legend()
# plt.title('Count of Olympians across the years')
# plt.show()

