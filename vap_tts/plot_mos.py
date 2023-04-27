import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/paper', help='Path to root directory')
    args = parser.parse_args()
    return args

def plot_predicted_mos(mos_pd):
    '''
    mos_pd: panda dataframe with columns 'tts', 'perm', 'mos'
    
    For each tts, plot the predicted MOS mean and std for each permutation in the same plot as meanand error bar plot, respectively.
    
    Normalize scale of MOS TO [1-5]
    '''
    tts_names = mos_pd['tts'].unique()
    tts_names = sorted(tts_names)
    perm_names = mos_pd['perm'].unique()
    perm_names = sorted(perm_names)

    fig, ax = plt.subplots(1, len(tts_names), figsize=(20, 5))
    for i, tts in enumerate(tts_names):
        mos_pd_tts = mos_pd[mos_pd['tts'] == tts]
        for perm in perm_names:
            mos_pd_tts_perm = mos_pd_tts[mos_pd_tts['perm'] == perm]
            mos = mos_pd_tts_perm['mos'].values
            mos_mean = np.mean(mos)
            mos_std = np.std(mos)
            # calculate 95% confidence interval
            mos_ci = 1.96 * mos_std / np.sqrt(len(mos))
            ax[i].bar(perm, mos_mean, yerr=mos_ci, label=perm)
        ax[i].set_xlabel('Utterance Type')
        ax[i].set_ylabel('Predicted MOS')
        ax[i].set_title(tts) 

        # normalize scale of MOS TO [1-5]
        ax[i].set_ylim([1, 5])

    return fig, ax

def main():
    args = parse_args()
    mos_pd = pd.read_pickle(os.path.join(args.root, "mos.pkl"))
    fig, ax = plot_predicted_mos(mos_pd)
    plt.savefig('mos_vis.png', dpi=600)

if __name__ == '__main__':
    main()