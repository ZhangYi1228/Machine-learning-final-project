import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
# For evaluation of the training progress:
# Check if historic data file exists, load it in or start from scratch.
FNAME_DATA = "Yi_Zhang-Changjing_Hu_agent_data.pt"
if os.path.isfile(FNAME_DATA):
    # Load historical training data.
    with open(FNAME_DATA, "rb") as file:
        historic_data = pickle.load(file)
    #game_nr = max(historic_data['games'])
    # Plot training progress every n:th game.
    #if game_nr != 0:
    # Incorporate the full training history.
    games_list = historic_data['games']
    #score_list = historic_data['score']
    coins_list = historic_data['coins']
    #crate_list = historic_data['crates']
    other_list = historic_data['enemies']




    score_list = np.array(coins_list) + 5*np.array(other_list)

    gamelist = []
    scorelist = np.split(score_list,100)
    scores = []
    sd = []
    for i in range(100):
        gamelist.append(i)
        scores.append(scorelist[i].sum())
        sd.append(scorelist[i].std())
    #explr_list = historic_data['exploration']

    # Plotting
    fig, ax = plt.subplots(2, figsize=(7.2, 5.4), sharex=True)

    # Mean score per 1000 game.
    ax[0].plot(gamelist, scores)
    ax[0].set_title('Mean score(coins + 5*kills) per 1000 rounds game')
    ax[0].set_ylabel('Mean score of per 1000 round')
    ax[0].set_xlabel('game round(*1000 round)')

    # sd per 1000 game.
    ax[1].plot(gamelist, sd)
    ax[1].set_title('standard deviation of per 1000 round')
    ax[1].set_ylabel('standard deviation of per 1000 round')
    ax[1].set_xlabel('game round(*1000 round)')


    # Export the figure.
    fig.tight_layout()
    plt.savefig(f'ModelEval.pdf')
    plt.close('all')

