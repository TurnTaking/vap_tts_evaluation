from argparse import ArgumentParser
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from vap_tts.utils import load_audio, load_text_grid, load_vap_output
from vap_tts.plot_utils import visualize_sample, visualize_human_sample
from vap_tts.results import plot_avg_prob_histogram

# from vap_tts.e_results import visualize_sample, get_args

st.set_page_config(layout="wide")

parser = ArgumentParser()
parser.add_argument("--root", type=str, default="data/multiwoz_tts_utts_dev")
args = parser.parse_args()


if __name__ == "__main__":

    if "df" not in st.session_state:
        df = pd.read_pickle(join(args.root, "results.pkl"))
        st.session_state.df = df
        st.session_state.tts = df.tts.unique()
        st.session_state.permutations = df.permutation.unique()
        st.session_state.result_sil_fig, _ = plot_avg_prob_histogram(df, plot=False)
        st.session_state.result_fig, _ = plot_avg_prob_histogram(
            df, silence=False, plot=False
        )

        # hdf = pd.read_pickle("data/human_baseline_vap.pkl")
        # st.session_state.hdf = hdf
        # st.session_state.hnames = hdf.name.to_list()

    tab1, tab2, tab3 = st.tabs(["TTS", "Human", "Result"])

    with tab1:
        tts_name = st.selectbox("TTS", st.session_state.tts)
        perm = st.selectbox("Permutation", st.session_state.permutations)

        df = st.session_state.df[st.session_state.df.tts == tts_name]
        df = df[df.permutation == perm]

        st.subheader("Sample index")
        ii = st.slider("n", 0, len(df) - 1, 0)

        d = df.iloc[ii]

        fig, _ = visualize_sample(d, args.root)
        st.pyplot(fig)
        st.audio(join(args.root, d["audio_path"]))

    # with tab2:
    #     # name = st.selectbox("Sample", st.session_state.hnames)
    #     st.header("Human samples")
    #     st.write("14, 84, 86")
    #     hdf = st.session_state.hdf
    #     ii = st.slider("n", 0, len(hdf) - 1, 0)
    #     d = hdf.iloc[ii]
    #     # hacky
    #     # d = st.session_state.hdf.loc[st.session_state.hdf.name == name].iloc[0]
    #     fig, _ = visualize_human_sample(d)
    #     st.pyplot(fig)
    #     st.audio(d.audio_path)

    with tab3:
        # st.pyplot(st.session_state.result_fig)
        # st.pyplot(st.session_state.result_sil_fig)

        st.subheader("Stats")
        cc1, cc2 = st.columns(2)
        cc1.text(
            """
        Ratio (n/total) over:
        * Shift:       P_now -> Shift P_future -> Shift
        * weak-shift:  P_now -> Shift P_future -> Hold
        * hold:        P_now -> Hold P_future -> Hold
        """
        )
        cc1.image("data/results/human_human_bar.png")
        cc2.image("data/results/stats.png")
        cc2.image("data/results/stats_tts.png")

        st.subheader("Histograms")
        st.text(
            """
            Histograms over "shift-volume". Values over 0 means a turn-yielding signal.
            Values to the right of the dashed line means that the user is expected to say something.

            In the Now plots it means that the user should say something short, a quick back and forth.
            In the Future plots it means that the user should really take the 'turn'.
            """
        )
        col1, col2 = st.columns(2)
        col1.subheader("Pause-Now")
        col1.image("data/results/pause_now_hist.png")
        col2.subheader("Pause-Future")
        col2.image("data/results/pause_fut_hist.png")

        col1.subheader("Shift-Now")
        col1.image("data/results/shift_now_hist.png")
        col2.subheader("Shift-Future")
        col2.image("data/results/shift_fut_hist.png")

        st.subheader("TTS")
        c1, c2 = st.columns(2)
        c1.subheader("Pause-Now")
        c1.image("data/results/pause_now_hist_AmazonTTS.png")
        c1.image("data/results/pause_now_hist_GoogleTTS.png")
        c1.image("data/results/pause_now_hist_MicrosoftTTS.png")
        c1.subheader("Shift-Now")
        c1.image("data/results/shift_now_hist_AmazonTTS.png")
        c1.image("data/results/shift_now_hist_GoogleTTS.png")
        c1.image("data/results/shift_now_hist_MicrosoftTTS.png")

        c2.subheader("Pause-Future")
        c2.image("data/results/pause_fut_hist_AmazonTTS.png")
        c2.image("data/results/pause_fut_hist_GoogleTTS.png")
        c2.image("data/results/pause_fut_hist_MicrosoftTTS.png")
        c2.subheader("Shift-Future")
        c2.image("data/results/shift_fut_hist_AmazonTTS.png")
        c2.image("data/results/shift_fut_hist_GoogleTTS.png")
        c2.image("data/results/shift_fut_hist_MicrosoftTTS.png")
