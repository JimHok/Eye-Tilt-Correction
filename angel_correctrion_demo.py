import streamlit as st
from PIL import Image
import random
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt

from module.iris_recog import *
from module.img_rotate import *
from module.img_process import *
from module.img_test import *
from module.method_test import *


st.markdown(
    "<h1 style='text-align: center;'>Eye Image Tilt Correction Demo</h1>",
    unsafe_allow_html=True,
)

with st.container():
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        img_num = st.number_input(
            "Image Number", min_value=0, max_value=999, step=1, value=10, format="%d"
        )
    with col2:
        img_side = st.selectbox("Image Side", ["L", "R"])
    with col3:
        img_take = st.number_input(
            "Image Take", min_value=0, max_value=9, step=1, value=1, format="%d"
        )

angle = st.number_input(
    "Set Angle", min_value=-90, max_value=90, step=1, value=0, format="%d"
)

st.write("")
with st.container():
    col1, col2 = st.columns([2, 2])
    with col1:
        button_iris = st.button(
            "Run Iris Demo", type="primary", use_container_width=True
        )
    with col2:
        button_eye = st.button("Run Eye Demo", type="primary", use_container_width=True)

    # with col2:
    #     button_m1 = st.button("Run Method 1", type="primary")
    # with col3:
    #     button_m2 = st.button("Run Method 2", type="primary")

with st.container():
    col1, col2 = st.columns([2, 2])
    with col1:
        button_iris_result = st.button(
            "Show Iris Test Result", type="primary", use_container_width=True
        )
    with col2:
        button_eye_result = st.button(
            "Show Eye Test Result", type="primary", use_container_width=True
        )

st.write("")
if button_iris:
    IrisProcessor(
        img_num, img_side, img_take, set_angle=angle, expand=True, plot=True, stlit=True
    ).process()

if button_eye:
    EyeProcessor(
        img_num,
        img_side,
        img_take,
        set_angle=angle,
        expand=False,
        plot=True,
        stlit=True,
    ).process()

# if button_m1:
#     method_1(img_num, img_side, img_take)

# if button_m2:
#     method_2(img_num, img_side, img_take, angle)

if button_iris_result:
    image = Image.open(f"output/iris_LR_500_hist_sep.png")
    st.image(image, caption="Test Result for 500 subjuects (L/R)")

    df_neg30 = pd.read_csv("temp_data/iris_test_neg30.csv")
    df_neg20 = pd.read_csv("temp_data/iris_test_neg20.csv")
    df_neg10 = pd.read_csv("temp_data/iris_test_neg10.csv")
    df_10 = pd.read_csv("temp_data/iris_test_10.csv")
    df_20 = pd.read_csv("temp_data/iris_test_20.csv")
    df_30 = pd.read_csv("temp_data/iris_test_30.csv")

    dfs = [df_neg30, df_neg20, df_neg10, df_10, df_20, df_30]
    mean = []
    median = []
    std_dev = []
    df_rot = ["-30", "-20", "-10", "10", "20", "30"]

    for i, df in enumerate(dfs):
        abs_diff = np.abs(df["Diff"])
        mean.append(round(abs_diff.mean(), 2))
        median.append(round(abs_diff.median(), 2))
        std_dev.append(round(abs_diff.std(), 2))

    results = {"Rotation": df_rot, "Mean": mean, "Median": median, "SD": std_dev}
    df_sd = pd.DataFrame(results)
    st.table(df_sd)

if button_eye_result:
    image = Image.open(f"output/eye_LR_500_hist_sep.png")
    st.image(image, caption="Test Result for 500 subjuects (L/R)")

    df_neg30 = pd.read_csv("temp_data/eye_test_neg30.csv")
    df_neg20 = pd.read_csv("temp_data/eye_test_neg20.csv")
    df_neg10 = pd.read_csv("temp_data/eye_test_neg10.csv")
    df_10 = pd.read_csv("temp_data/eye_test_10.csv")
    df_20 = pd.read_csv("temp_data/eye_test_20.csv")
    df_30 = pd.read_csv("temp_data/eye_test_30.csv")

    dfs = [df_neg30, df_neg20, df_neg10, df_10, df_20, df_30]
    mean = []
    median = []
    std_dev = []
    df_rot = ["-30", "-20", "-10", "10", "20", "30"]

    for i, df in enumerate(dfs):
        abs_diff = np.abs(df["Diff"])
        mean.append(round(abs_diff.mean(), 2))
        median.append(round(abs_diff.median(), 2))
        std_dev.append(round(abs_diff.std(), 2))

    results = {"Rotation": df_rot, "Mean": mean, "Median": median, "SD": std_dev}
    df_sd = pd.DataFrame(results)
    st.table(df_sd)
