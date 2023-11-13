import multiprocessing as mp
from module.img_rotate import *
import streamlit as st


def plot_results(imgs, in_circles, out_circles, templates, angle, angle_counter, stlit=False):
    fig = plt.figure(figsize=(20, 10*1), constrained_layout=False)
    outer_grid = fig.add_gridspec(1, 3, wspace=0.1, hspace=-0.2)
    title = ['Original Image',
             f'Rotated Image ({angle}°)', f'Counter Rotated Image ({angle_counter}°)']

    for i in range(len(imgs)):
        img = imgs[i]

        inner_grid = outer_grid[0, i].subgridspec(4, 1, hspace=-0.75)
        ax0 = fig.add_subplot(inner_grid[0:2])
        ax1 = fig.add_subplot(inner_grid[3])

        ax0.imshow(img, cmap='gray')

        if i < len(out_circles):
            if len(out_circles[0]) == 3:
                ax0.set_title(title[i])
                in_circle = plt.Circle(
                    (in_circles[i][0], in_circles[i][1]), in_circles[i][2], color='g', fill=False, linewidth=1)
                ax0.add_patch(in_circle)
                ax0.scatter(
                    in_circles[i][0], in_circles[i][1], s=20, c='g', marker='o')

                out_circle = plt.Circle(
                    (out_circles[i][0], out_circles[i][1]), out_circles[i][2], color='b', fill=False, linewidth=1)
                ax0.add_patch(out_circle)
                ax0.scatter(
                    out_circles[i][0], out_circles[i][1], s=20, c='b', marker='o')

                ax1.imshow(templates[i], cmap='gray')
                ax1.set_title(f'Binary Encoded Image')
            else:
                ax0.plot(out_circles[i][:, 1],
                         out_circles[i][:, 0], '-b', lw=1)
                ax0.set_title(title[i])

                if in_circles[i][2] is None:
                    print(f"No circles found in image {i}")
                    ax1.imshow(img, cmap='gray')
                    ax1.axis([0, 800, 64, 0])
                else:
                    circle = plt.Circle(
                        (in_circles[i][0], in_circles[i][1]), in_circles[i][2], color='g', fill=False, linewidth=1)
                    ax0.add_patch(circle)
                    ax0.scatter(
                        in_circles[i][0], in_circles[i][1], s=20, c='g', marker='o')

                    ax1.imshow(templates[i], cmap='gray')
                    ax1.set_title(f'Binary Encoded Image')

        else:
            ax0.set_title(title[i])
            ax1.imshow(img, cmap='gray')
            ax1.axis([0, 800, 64, 0])

    if stlit:
        st.pyplot(fig)
    else:
        plt.show()


def run_test_df(function, img_nums, img_sides, img_takes, img_width=400, set_angle=None):
    hd_raws = []
    angles = []
    shifts = []

    for img_num in tqdm(range(img_nums)):
        for img_side in img_sides:
            for img_take in range(img_takes):
                hd_raw, angle, shift = function(
                    img_num, img_side, img_take, set_angle).process()
                hd_raws.append(hd_raw)
                angles.append(angle)
                shifts.append(shift)

    df = pd.DataFrame({'Hamming Distance': hd_raws, 'Rotate': angles,
                      'Shift': shifts, 'Counter Rotate': round(360 / img_width, 2) * np.array(shifts)})
    df['Diff'] = df['Rotate'] + df['Counter Rotate']

    return df


def run_test_df_multi(function, img_nums, img_sides, img_takes, img_width=400, set_angle=None, process=8):
    hd_raws = []
    angles = []
    shifts = []

    with mp.Pool(processes=process) as pool:
        results = []
        for img_num in range(img_nums):
            for img_side in img_sides:
                for img_take in range(img_takes):
                    results.append(pool.apply_async(
                        function, (img_num, img_side, img_take, set_angle)))

        for result in tqdm(results):
            hd_raw, angle, shift = result.get().process()
            hd_raws.append(hd_raw)
            angles.append(angle)
            shifts.append(shift)

    df = pd.DataFrame({'Hamming Distance': hd_raws, 'Rotate': angles,
                      'Shift': shifts, 'Counter Rotate': round(360 / img_width, 2) * np.array(shifts)})
    df['Diff'] = df['Rotate'] + df['Counter Rotate']

    return df