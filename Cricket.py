import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np

iterations = 1000

# Load the tables from CSV files with an alternative encod ing
try:
    ratings = pd.read_csv(r"C:/Users/manas/Desktop/Manas/Coding/Projects/Cricket/Data_set.csv", encoding='utf-8')
except UnicodeDecodeError:
    ratings = pd.read_csv(r"C:/Users/manas/Desktop/Manas/Coding/Projects/Cricket/Data_set.csv", encoding='latin1')

try:
    win_prob = pd.read_csv(r"C:/Users/manas/Desktop/Manas/Coding/Projects/Cricket/Data_set - Copy.csv", encoding='utf-8')
except UnicodeDecodeError:
    win_prob = pd.read_csv(r"C:/Users/manas/Desktop/Manas/Coding/Projects/Cricket/Data_set - Copy.csv", encoding='latin1')

# Initialize dictionary to record final winners
final_winner_counts = {}
def adjust_win_probability(rating_diff):
    # Define a scaling factor for rating difference
    k = 0.015
    # Adjust win probabilities using a logistic function
    adjusted_win_prob1 = 1 / (1 + np.exp(-k * rating_diff))
    adjusted_win_prob2 = 1 - adjusted_win_prob1  # Ensure probabilities sum to 1
    return adjusted_win_prob1, adjusted_win_prob2

# Define a logistic function to adjust win probabilities based on rating difference
def game(df, team1, team2, adjusted_probs=None):
    team1_row = df[df['country'] == team1]
    team2_row = df[df['country'] == team2]
    rating1 = team1_row['rating'].values[0]
    rating2 = team2_row['rating'].values[0]

    if adjusted_probs is None:
        # Calculate the rating difference
        rating_diff = rating1 - rating2
        # Adjust win probabilities based on rating difference
        adjusted_win_prob1, adjusted_win_prob2 = adjust_win_probability(rating_diff)
        adjusted_probs = (adjusted_win_prob1, adjusted_win_prob2)

    # print(f"Adjusted win_prob1: {adjusted_win_prob1}")
    # print(f"Adjusted win_prob2: {adjusted_win_prob2}")
    # print(f"Rating difference: {rating1 - rating2}")

    # Set thresholds based on the probabilities
    threshold_team1 = adjusted_win_prob1 * 100
    threshold_team2 = adjusted_win_prob2 * 100

    # Generate random numbers between 0 and 99
    random_number1 = random.randint(0, 99)
    random_number2 = random.randint(0, 99)

    # Determine the result for each team based on thresholds and random numbers
    result1 = 1 if random_number1 < threshold_team1 else 0
    result2 = 1 if random_number2 < threshold_team2 else 0

    # Compare results and return the winning team
    if result2 > result1:
        return team2
    elif result1 > result2:
        return team1
    else:
        return game(df, team1, team2)  


# Combine the DataFrames vertically
combined_df = pd.concat([ratings, win_prob], ignore_index=True)

# Group by country and aggregate ratings and winning probabilities
data_df = combined_df.groupby('country').agg({'rating': 'max', 'win_prob': 'max'}).reset_index()

# print(game(data_df, 'india', 'england'))
print(game(data_df, 'england', 'india'))

for _ in range(iterations):
    countries = [
        'afghanistan', 'australia', 'bangladesh', 'canada', 'england',
        'india', 'ireland', 'namibia', 'netherlands', 'nepal', 'new zealand',
        'oman', 'pakistan', 'papua new guinea', 'scotland', 'south africa',
        'sri lanka', 'uganda', 'usa', 'west indies'
    ]

    ratings = ratings[ratings['country'].isin(countries)]
    win_prob = win_prob[win_prob['country'].isin(countries)]

    # Combine the DataFrames vertically
    combined_df = pd.concat([ratings, win_prob], ignore_index=True)

    # Group by country and aggregate ratings and winning probabilities
    data_df = combined_df.groupby('country').agg({'rating': 'max', 'win_prob': 'max'}).reset_index()

    matches = [
        ['usa', 'canada'],
        ['west indies', 'papua new guinea'],
        ['namibia', 'oman'],
        ['sri lanka', 'south africa'],
        ['afghanistan', 'uganda'],
        ['england', 'scotland'],
        ['netherlands', 'nepal'],
        ['india', 'ireland'],
        ['papua new guinea', 'uganda'],
        ['australia', 'oman'],
        ['usa', 'pakistan'],
        ['namibia', 'scotland'],
        ['canada', 'ireland'],
        ['new zealand', 'afghanistan'],
        ['sri lanka', 'bangladesh'],
        ['netherlands', 'south africa'],
        ['australia', 'england'],
        ['west indies', 'uganda'],
        ['india', 'pakistan'],
        ['oman', 'scotland'],
        ['south africa', 'bangladesh'],
        ['pakistan', 'canada'],
        ['sri lanka', 'nepal'],
        ['australia', 'namibia'],
        ['usa', 'india'],
        ['west indies', 'new zealand'],
        ['england', 'oman'],
        ['bangladesh', 'netherlands'],
        ['afghanistan', 'papua new guinea'],
        ['usa', 'ireland'],
        ['south africa', 'nepal'],
        ['new zealand', 'uganda'],
        ['india', 'canada'],
        ['namibia', 'england'],
        ['australia', 'scotland'],
        ['pakistan', 'ireland'],
        ['bangladesh', 'nepal'],
        ['sri lanka', 'netherlands'],
        ['new zealand', 'papua new guinea'],
        ['west indies', 'afghanistan']
    ]

    winners = []
    for match in matches:
        winners.append(game(data_df, match[0], match[1]))
    def sorter_by_rank(group):
        for i, team in enumerate(group):
            # Check if this team has the same win count as the next team
            if i < len(group) - 1 and team[1:] == group[i + 1][1:]:
                # Get the ratings of the current team and the next team
                rating_current = data_df[data_df['country'] == team[0]]['rating'].values[0]
                rating_next = data_df[data_df['country'] == group[i + 1][0]]['rating'].values[0]
                # Compare the ratings
                if rating_next > rating_current:
                    # Swap the positions of the current team and the next team
                    group[i], group[i + 1] = group[i + 1], group[i]
        return group

    group1 = [
        ['usa', winners.count('usa')], ['canada', winners.count('canada')], ['india', winners.count('india')], ['ireland', winners.count('ireland')], ['pakistan', winners.count('pakistan')]
    ]
    group1 = sorted(group1, key=lambda x: x[1], reverse=True)

    group2 = [
        ['namibia', winners.count('namibia')], ['england', winners.count('england')], ['scotland', winners.count('scotland')], ['oman', winners.count('oman')], ['australia', winners.count('australia')]
    ]
    group2 = sorted(group2, key=lambda x: x[1], reverse=True)
    group3 = [
        ['afghanistan', winners.count('afghanistan')], ['west indies', winners.count('west indies')], ['papua new guinea', winners.count('papua new guinea')], ['uganda', winners.count('uganda')], ['new zealand', winners.count('new zealand')]
    ]
    group3 = sorted(group3, key=lambda x: x[1], reverse=True)
    group4 = [
        ['south africa', winners.count('south africa')], ['netherlands', winners.count('netherlands')], ['nepal', winners.count('nepal')], ['sri lanka', winners.count('sri lanka')], ['bangladesh', winners.count('bangladesh')]
    ]
    group4 = sorted(group4, key=lambda x: x[1], reverse=True)
    
    group1 = sorter_by_rank(group1)
    group2 = sorter_by_rank(group2)
    group3 = sorter_by_rank(group3)
    group4 = sorter_by_rank(group4)

    group1_Q = group1[:2]
    group2_Q = group2[:2]
    group3_Q = group3[:2]
    group4_Q = group4[:2]

    super_8 = [
        [group1_Q[1][0], group4_Q[0][0]],  # A2 vs D1
        [group2_Q[0][0], group3_Q[1][0]],  # B1 vs C2
        [group3_Q[0][0], group1_Q[0][0]],  # C1 vs A1
        [group2_Q[1][0], group4_Q[1][0]],  # B2 vs D2
        [group2_Q[0][0], group4_Q[0][0]],  # B1 vs D1
        [group1_Q[1][0], group3_Q[1][0]],  # A2 vs C2
        [group1_Q[0][0], group4_Q[1][0]],  # A1 vs D2
        [group3_Q[0][0], group2_Q[1][0]],  # C1 vs B2
        [group1_Q[1][0], group2_Q[0][0]],  # A2 vs B1
        [group3_Q[1][0], group4_Q[0][0]],  # C2 vs D1
        [group2_Q[1][0], group1_Q[0][0]],  # B2 vs A1
        [group3_Q[0][0], group4[1][0]],  # C1 vs D2
    ]

    winners_super_8 = []
    for match in super_8:
        winner = game(data_df, match[0], match[1])
        winners_super_8.append(winner)
    # Count the wins for each team in the super 8 phase
    super_8_win_counts = {}
    for team in winners_super_8:
        super_8_win_counts[team] = winners_super_8.count(team)

    # Sort the teams based on the number of wins
    sorted_super_8_teams = sorted(super_8_win_counts.items(), key=lambda x: x[1], reverse=True)
    sorter_by_rank(sorted_super_8_teams)
    # print(sorted_super_8_teams)
    # # Get the top 4 teams
    top_4_teams = sorted_super_8_teams[:4]
    # print(top_4_teams)

#     # Semi-finals
    semi_1 = [top_4_teams[0][0], top_4_teams[1][0]]
    semi_1_winner = game(data_df, semi_1[0], semi_1[1])
    semi_2 = [top_4_teams[2][0], top_4_teams[3][0]]
    semi_2_winner = game(data_df, semi_2[0], semi_2[1])
    # Final
    final = [semi_1_winner, semi_2_winner]
    final_winner = game(data_df, final[0], final[1])
    # Record the final winner
    if final_winner in final_winner_counts:
        final_winner_counts[final_winner] += 1
    else:
        final_winner_counts[final_winner] = 1

# Print the final winner counts
print(f"Final winner counts after {iterations} simulations:")
final_winner_counts = dict(sorted(final_winner_counts.items(), key=lambda item: item[1], reverse=True))
print(final_winner_counts)
final_winner_team = list(final_winner_counts.keys())
final_winner_team.reverse()
final_winner_percentage = []
for value in final_winner_counts.values():
    final_winner_percentage.append(value/iterations)

final_winner_percentage = sorted(final_winner_percentage)


cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0, 1, len(final_winner_team)))

fig = plt.figure(figsize=(20, 15))
# # creating the bar plot
plt.bar(final_winner_team, final_winner_percentage, color=colors, width=0.8)

plt.xlabel("Teams")
plt.ylabel("Win probability")
plt.title(f"Win probability of the ICC T20 World Cup: iterations = {iterations}")
plt.xticks(rotation=90)
plt.show()
