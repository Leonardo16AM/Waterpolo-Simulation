import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.interpolate import make_interp_spline

df = pd.read_csv('water_polo_results.csv')


T = 1 
num_points = 480
t_points = np.linspace(0, T, num_points)

control_points = np.linspace(0, T, 6)
SPLINE_VALUES = np.array([1, 2, 1.9, 2, 2, 1]) 


def lambda_function(t, control_points, values):
    spline = make_interp_spline(control_points, values, k=3)
    return spline(t)

def simulate_goals(lambda_function, T, control_points, values, average_goals, num_points=100):
    t_points = np.linspace(0, T, num_points)
    lambda_points = lambda_function(t_points, control_points, values)
    lambda_points = lambda_points / np.sum(lambda_points) * average_goals
    goals = np.random.poisson(lambda_points)
    return np.sum(goals)

def simulate_match(team1, team2, df):
    games = df[((df['Team1'] == team1) & (df['Team2'] == team2)) | ((df['Team1'] == team2) & (df['Team2'] == team1))]

    q1_t1, q2_t1, q3_t1, q4_t1 = [], [], [], []
    q1_t2, q2_t2, q3_t2, q4_t2 = [], [], [], []

    for i in range(len(games)):
        if games.iloc[i]['Team1'] == team1:
            q1_t1.append(games.iloc[i]['q1_A'])
            q2_t1.append(games.iloc[i]['q2_A'])
            q3_t1.append(games.iloc[i]['q3_A'])
            q4_t1.append(games.iloc[i]['q4_A'])
            q1_t2.append(games.iloc[i]['q1_B'])
            q2_t2.append(games.iloc[i]['q2_B'])
            q3_t2.append(games.iloc[i]['q3_B'])
            q4_t2.append(games.iloc[i]['q4_B'])
        if games.iloc[i]['Team1'] == team2:
            q1_t2.append(games.iloc[i]['q1_A'])
            q2_t2.append(games.iloc[i]['q2_A'])
            q3_t2.append(games.iloc[i]['q3_A'])
            q4_t2.append(games.iloc[i]['q4_A'])
            q1_t1.append(games.iloc[i]['q1_B'])
            q2_t1.append(games.iloc[i]['q2_B'])
            q3_t1.append(games.iloc[i]['q3_B'])
            q4_t1.append(games.iloc[i]['q4_B'])

    if len(q1_t1) == 0 and len(q1_t2) == 0:
        return team1


    avg_q1_t1 = np.mean(q1_t1) if q1_t1 else 0
    avg_q2_t1 = np.mean(q2_t1) if q2_t1 else 0
    avg_q3_t1 = np.mean(q3_t1) if q3_t1 else 0
    avg_q4_t1 = np.mean(q4_t1) if q4_t1 else 0

    avg_q1_t2 = np.mean(q1_t2) if q1_t2 else 0
    avg_q2_t2 = np.mean(q2_t2) if q2_t2 else 0
    avg_q3_t2 = np.mean(q3_t2) if q3_t2 else 0
    avg_q4_t2 = np.mean(q4_t2) if q4_t2 else 0

    s1 = (
        simulate_goals(lambda_function, T, control_points, SPLINE_VALUES, avg_q1_t1) +
        simulate_goals(lambda_function, T, control_points, SPLINE_VALUES, avg_q2_t1) +
        simulate_goals(lambda_function, T, control_points, SPLINE_VALUES, avg_q3_t1) +
        simulate_goals(lambda_function, T, control_points, SPLINE_VALUES, avg_q4_t1)
    )
    s2 = (
        simulate_goals(lambda_function, T, control_points, SPLINE_VALUES, avg_q1_t2) +
        simulate_goals(lambda_function, T, control_points, SPLINE_VALUES, avg_q2_t2) +
        simulate_goals(lambda_function, T, control_points, SPLINE_VALUES, avg_q3_t2) +
        simulate_goals(lambda_function, T, control_points, SPLINE_VALUES, avg_q4_t2)
    )

    return team1 if s1 > s2 else team2


def simulate_group_stage(teams):
    points = {team: 0 for team in teams}
    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i < j:
                winner = simulate_match(team1, team2,df)
                points[winner] += 2
    return points

def rank_teams(points):
    return sorted(points.items(), key=lambda x: (-x[1], x[0]))

def simulate_knockout_stage(teams):
    winners = []
    for i in range(0, len(teams), 2):
        winner = simulate_match(teams[i], teams[i+1],df)
        winners.append(winner)
    return winners

def simulate_tournament(MENS_POOLS):
    group_results = {}
    for group, teams in MENS_POOLS.items():
        points = simulate_group_stage(teams)
        ranked_teams = rank_teams(points)
        group_results[group] = [team for team, _ in ranked_teams]
    
    quarter_finalists_A = group_results['A'][:4]
    quarter_finalists_B = group_results['B'][:4]

    quarter_finals = [
        (quarter_finalists_A[0], quarter_finalists_B[3]),
        (quarter_finalists_A[1], quarter_finalists_B[2]),
        (quarter_finalists_A[2], quarter_finalists_B[1]),
        (quarter_finalists_A[3], quarter_finalists_B[0]),
    ]
    
    quarter_final_winners = []
    for match in quarter_finals:
        winner = simulate_match(match[0], match[1],df)
        quarter_final_winners.append(winner)
    
    semi_finals = [
        (quarter_final_winners[0], quarter_final_winners[1]),
        (quarter_final_winners[2], quarter_final_winners[3]),
    ]
    
    semi_final_winners = []
    semi_final_losers = []
    for match in semi_finals:
        winner = simulate_match(match[0], match[1],df)
        loser = match[0] if winner == match[1] else match[1]
        semi_final_winners.append(winner)
        semi_final_losers.append(loser)
    
    gold_medal_winner = simulate_match(semi_final_winners[0], semi_final_winners[1],df)
    silver_medal_winner = semi_final_winners[1] if gold_medal_winner == semi_final_winners[0] else semi_final_winners[0]
    
    bronze_medal_winner = simulate_match(semi_final_losers[0], semi_final_losers[1],df)
    
    return {
        'gold': gold_medal_winner,
        'silver': silver_medal_winner,
        'bronze': bronze_medal_winner
    }





st.title('Simulador de Torneo de Polo Acuático')

team_list = ['Argentina', 'Australia', 'Brazil', 'Canada', 'Croatia', 'France',
             'Georgia', 'Germany', 'Greece', 'Hungary', 'Italy', 'Japan',
             'Kazakhstan', 'Montenegro', "People's Republic of China",
             'Romania', 'Serbia', 'South Africa', 'Spain', 'United States of America']


group_a = st.multiselect('Grupo A:', team_list, default=['Croatia', 'Italy', 'Romania'])
group_b = st.multiselect('Grupo B:', team_list, default=['France', 'Hungary', 'Spain'])

with st.sidebar:
    st.title("Configuración de la Simulación")
    spline_values = [st.slider(f'Valor {i} para la Spline:', 0.0, 5.0, 1.0, step=0.1, format="%.2f") for i in range(1, 7)]
    num_simulations = st.number_input('Número de simulaciones:', min_value=1, max_value=1000, value=100, step=1)
    start_prediction = st.button('Start Prediction')

control_points = np.linspace(0, 1, 6)  
spline = make_interp_spline(control_points, spline_values, k=3)
x = np.linspace(0, 1, 300)
y = spline(x)

sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 4))
plt.plot(x, y, color='red', label='Spline Interpolada') 
plt.scatter(control_points, spline_values, color='red', zorder=5) 
plt.title('Visualización del Spline')
plt.xlabel('Tiempo Normalizado')
plt.ylabel('Valor del Spline')
plt.legend()
st.pyplot(plt.gcf())

if start_prediction:
    MENS_POOLS = {'A': group_a, 'B': group_b}
    results = []
    for _ in range(num_simulations):
        results.append(simulate_tournament(MENS_POOLS))
    
    medals = {}
    for result in results:
        for medal, team in result.items():
            if team not in medals:
                medals[team] = {'gold': 0, 'silver': 0, 'bronze': 0}
            medals[team][medal] += 1

    sorted_teams = sorted(medals.items(), key=lambda item: (-item[1]['gold'], -item[1]['silver'], -item[1]['bronze']))

    data = {team: [medal_counts['gold'], medal_counts['silver'], medal_counts['bronze']] for team, medal_counts in sorted_teams}
    df = pd.DataFrame(data, index=['Gold', 'Silver', 'Bronze']).T
    st.write(df)
    