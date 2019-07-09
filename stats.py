import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Team import Team

if __name__ == "__main__":
    df = pd.read_csv('answers.csv').drop(['Отметка времени'], axis=1)

    column_names = df.columns
    teams_arr = []
    teams = {}

    for col in df:
        T = Team(col, np.array(df[col]))
        teams[col] = T
        teams_arr.append(T)

    means, medians = [], []
    for team in teams_arr:
        means.append(team.mean_rate)
        medians.append(team.median_rate)

    means_idx = np.argsort(means)
    medians_idx = np.argsort(medians)

    column_names = np.array(column_names)
    means = np.array(means)
    medians = np.array(medians)

    plt.figure(figsize=(20, 8))
    bars = plt.bar(column_names[means_idx], means[means_idx])
    plt.title('Отсортированные по средним оценкам результаты')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.3, yval + .05, round(yval, 2))
    plt.tight_layout()
    plt.savefig('means/means.png')
    plt.clf()

    col_names = ['Тур', 'Дата', 'Дома', 'Гости']
    schedule = pd.read_csv('schedule.csv', sep=' ', header=None, names=col_names).drop(['Дата'], axis=1)
    schedule = schedule.replace('Арсенал', 'Арсенал Тула')
    schedule = schedule.replace('Крылья_Советов', 'Крылья Советов')

    team_matches = {}
    team_HA = {}

    for team in column_names:
        team_matches[team] = []
        team_HA[team] = []

    for index, row in schedule.iterrows():
        team_matches[row['Дома']].append(teams[row['Гости']])
        team_matches[row['Гости']].append(teams[row['Дома']])

        team_HA[row['Дома']].append('H')
        team_HA[row['Гости']].append('A')

    #f, axarr = plt.subplots(16, 1, figsize=(20, 100))
    for i, team in enumerate(column_names):
        plt.figure(figsize=(20, 8))
        threshold = teams[team].mean_rate
        opp_means, opp_names = [], []
        for j, opponent in enumerate(team_matches[team][:9]):
            opp_means.append(opponent.mean_rate)
            opp_names.append(opponent.name + ' (' + team_HA[team][j] + ')')
        bars = plt.bar(opp_names, opp_means)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + 0.3, yval + .05, round(threshold - yval, 1))
        plt.plot([-1., 9.], [threshold, threshold], "k-")
        plt.title(team)
        plt.tight_layout()
        plt.savefig('no_ha_pers/' + str(team) + '.png')
        plt.clf()


    #учет H/A
    col_names = ['Команда', 'Дома', 'Гости']
    ha = pd.read_csv('ha.csv', sep=' ', header=None, names=col_names)
    ha = ha.replace('Арсенал_Тула', 'Арсенал Тула')
    ha = ha.replace('Крылья_Советов', 'Крылья Советов')

    for index, row in ha.iterrows():
        coef = row['Гости'] / row['Дома']
        teams[row['Команда']].RH = 2 * teams[row['Команда']].mean_rate / (1.0 + coef)
        teams[row['Команда']].RA = teams[row['Команда']].RH * coef

    means_H, means_A, team_names = [], [], []

    for team in teams:
        means_H.append(teams[team].RH)
        means_A.append(teams[team].RA)
        team_names.append(teams[team].name)

    meansH_idx = np.argsort(means_H)
    meansA_idx = np.argsort(means_A)

    means_H = np.array(means_H)
    means_A = np.array(means_A)
    team_names = np.array(team_names)

    plt.figure(figsize=(20, 8))
    bars = plt.bar(team_names[meansH_idx], means_H[meansH_idx])
    plt.title('Отсортированные по средним ДОМАШНИМ оценкам результаты')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.3, yval + .05, round(yval, 2))
    plt.tight_layout()
    plt.savefig('means/h_means.png')
    plt.clf()

    plt.figure(figsize=(20, 8))
    bars = plt.bar(team_names[meansA_idx], means_A[meansA_idx])
    plt.title('Отсортированные по средним ГОСТЕВЫМ оценкам результаты')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.3, yval + .05, round(yval, 2))
    plt.tight_layout()
    plt.savefig('means/a_means.png')
    plt.clf()

    first_three, first, second, third = {}, {}, {}, {}
    for team in column_names:
        first_three[team], first[team], second[team], third[team] = 0, 0, 0, 0

        plt.figure(figsize=(20, 8))
    for i, team in enumerate(column_names):
        threshold = []
        opp_means, opp_names = [], []
        for j, opponent in enumerate(team_matches[team][:9]):
            if team_HA[team][j] == 'H':
                opp_means.append(opponent.RA)
                threshold.append(teams[team].RH)
            else:
                opp_means.append(opponent.RH)
                threshold.append(teams[team].RA)

            opp_names.append(opponent.name + ' (' + team_HA[team][j] + ')')
        bars = plt.bar(opp_names, opp_means)
        for j, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + 0.3, yval + .05, round(threshold[j] - yval, 1))
            if j < 3:
                first_three[team] += round(threshold[j] - yval, 1)
            if j == 0:
                first[team] = round(threshold[j] - yval, 1)
            elif j == 1:
                second[team] = round(threshold[j] - yval, 1)
            elif j == 2:
                third[team] = round(threshold[j] - yval, 1)

        plt.plot(range(0, 9), threshold, "k-")
        plt.title(team)
        plt.tight_layout()
        plt.savefig('ha/' + str(team) + '.png')
        plt.clf()

    first_three_arr = []
    for team in column_names:
        first_three_arr.append(first_three[team])

    first_three_arr = np.array(first_three_arr)
    idx_ff = np.argsort(first_three_arr)
    ff_cols = column_names[idx_ff][::-1]

    with open('out_stats.txt', 'w') as the_file:
        print('Первые три тура:', file=the_file)
        for i, val in enumerate(first_three_arr[idx_ff][:9:-1]):
            print(ff_cols[i], round(val, 1), file=the_file)

        first_arr = []
        for team in column_names:
            first_arr.append(first[team])

        first_arr = np.array(first_arr)
        idx_f = np.argsort(first_arr)
        f_cols = column_names[idx_f][::-1]

        print(file=the_file)
        print('1 тур:', file=the_file)
        for i, val in enumerate(first_arr[idx_f][:9:-1]):
            print(f_cols[i], round(val, 1), file=the_file)

        second_arr = []
        for team in column_names:
            second_arr.append(second[team])

        second_arr = np.array(second_arr)
        idx_s = np.argsort(second_arr)
        s_cols = column_names[idx_s][::-1]

        print(file=the_file)
        print('2 тур:', file=the_file)
        for i, val in enumerate(second_arr[idx_s][:9:-1]):
            print(s_cols[i], round(val, 1), file=the_file)

        third_arr = []
        for team in column_names:
            third_arr.append(third[team])

        third_arr = np.array(third_arr)
        idx_t = np.argsort(third_arr)
        t_cols = column_names[idx_t][::-1]

        print(file=the_file)
        print('3 тур:', file=the_file)
        for i, val in enumerate(third_arr[idx_t][:9:-1]):
            print(t_cols[i], round(val, 1), file=the_file)