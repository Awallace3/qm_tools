import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

kcal_per_mol = "$\mathrm{kcal\cdot mol^{-1}}$"


def create_minor_y_ticks(ylim):
    diff = abs(ylim[1] - ylim[0])
    if diff > 100:
        inc = 10
    if diff > 20:
        inc = 5
    elif diff > 10:
        inc = 2.5
    else:
        inc = 1
    lower_bound = int(ylim[0])
    while lower_bound % inc != 0:
        lower_bound -= 1
    upper_bound = int(ylim[1])
    while upper_bound % inc != 0:
        upper_bound += 1
    upper_bound += inc
    minor_yticks = np.arange(lower_bound, upper_bound, inc)
    return minor_yticks


def violin_plot(
    df,
    df_labels_and_columns: {},
    output_filename: str,
    plt_title: str = None,
    output_path: str = "./",
    bottom: float = 0.4,
    ylim: list = None,
    transparent: bool = False,
    widths: float = 0.85,
    figure_size: tuple = None,
    set_xlable=False,
    x_label_rotation=90,
    x_label_fontsize=8,
    ylabel=r"Error ($\mathrm{kcal\cdot mol^{-1}}$)",
    dpi=600,
    usetex=True,
    rcParams={
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
    },
    colors: list = None,
) -> None:
    """
    Create a dataframe with columns of errors pre-computed for generating
    violin plots with MAE, RMSE, and MaxAE displayed above each violin.

    Args:
        df: DataFrame with columns of errors 
        df_labels_and_columns: Dictionary of plotted labels along with the df column for data
        output_filename: Name of the output file
        ylim: list =[-15, 35],
        rcParams: can be set to None if latex is not used
        colors: list of colors for each df column plotted. A default will alternate between blue and green.
    """
    print(f"Plotting {output_filename}")
    if rcParams is not None:
        plt.rcParams.update(rcParams)
    vLabels, vData = [], []
    annotations = []  # [(x, y, text), ...]
    cnt = 1
    plt.rcParams["text.usetex"] = usetex
    for k, v in df_labels_and_columns.items():
        df[v] = pd.to_numeric(df[v])
        df_sub = df[df[v].notna()].copy()
        vData.append(df_sub[v].to_list())
        k_label = "\\textbf{" + k + "}"
        vLabels.append(k_label)
        m = df_sub[v].max()
        rmse = df_sub[v].apply(lambda x: x**2).mean() ** 0.5
        mae = df_sub[v].apply(lambda x: abs(x)).mean()
        max_error = df_sub[v].apply(lambda x: abs(x)).max()
        text = r"\textit{%.2f}" % mae
        text += "\n"
        text += r"\textbf{%.2f}" % rmse
        text += "\n"
        text += r"\textrm{%.2f}" % max_error
        annotations.append((cnt, m, text))
        cnt += 1

    pd.set_option("display.max_columns", None)
    fig = plt.figure(dpi=dpi)
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    ax = plt.subplot(111)
    vplot = ax.violinplot(
        vData,
        showmeans=True,
        showmedians=False,
        quantiles=[[0.05, 0.95] for i in range(len(vData))],
        widths=widths,
    )
    for n, partname in enumerate(["cbars", "cmins", "cmaxes", "cmeans"]):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)
    quantile_color = "red"
    quantile_style = "-"
    quantile_linewidth = 0.8
    for n, partname in enumerate(["cquantiles"]):
        vp = vplot[partname]
        vp.set_edgecolor(quantile_color)
        vp.set_linewidth(quantile_linewidth)
        vp.set_linestyle(quantile_style)
        vp.set_alpha(1)

    colors = ["blue" if i % 2 == 0 else "green" for i in range(len(vLabels))]
    for n, pc in enumerate(vplot["bodies"], 1):
        pc.set_facecolor(colors[n - 1])
        pc.set_alpha(0.6)

    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for i in range(len(xs_error))],
        "k--",
        label=r"$\pm$1 $\mathrm{kcal\cdot mol^{-1}}$",
        zorder=0,
        linewidth=0.6,
    )
    ax.plot(
        xs_error,
        [0 for i in range(len(xs_error))],
        "k--",
        linewidth=0.5,
        alpha=0.5,
        # label=r"Reference Energy",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [-1 for i in range(len(xs_error))],
        "k--",
        zorder=0,
        linewidth=0.6,
    )
    ax.plot(
        [],
        [],
        linestyle=quantile_style,
        color=quantile_color,
        linewidth=quantile_linewidth,
        label=r"5-95th Percentile",
    )
    navy_blue = (0.0, 0.32, 0.96)
    ax.set_xticks(xs)
    plt.setp(
        ax.set_xticklabels(vLabels),
        rotation=x_label_rotation,
        fontsize=x_label_fontsize,
    )
    ax.set_xlim((0, len(vLabels)))
    if ylim is not None:
        ax.set_ylim(ylim)
        minor_yticks = create_minor_y_ticks(ylim)
        ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc="upper left", edgecolor="black", fontsize="8")

    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k")
    ax.set_ylabel(ylabel, color="k")

    ax.grid(color="#54585A", which="major", linewidth=0.5, alpha=0.5, axis="y")
    ax.grid(color="#54585A", which="minor", linewidth=0.5, alpha=0.5)
    # Annotations of RMSE
    for x, y, text in annotations:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x, y + 0.1),
            color="black",
            fontsize="8",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(colors[n - 1])
        xtick.set_alpha(0.8)

    if plt_title is not None:
        plt.title(f"{plt_title}")
    fig.subplots_adjust(bottom=bottom)
    ext = "png"
    if len(output_filename.split(".")) > 1:
        ext = output_filename.split(".")[-1]
    output_basename = os.path.basename(output_filename)
    plt.savefig(
        f"{output_path}/{output_basename}_violin.{ext}",
        transparent=transparent,
        bbox_inches="tight",
        dpi=dpi,
    )
    plt.clf()
    return


if __name__ == "__main__":
    # Fake data generated for example
    n_samples = 1000
    mean = 0.5  # replace with your desired mean
    std_dev = 5  # replace with your desired standard deviation
    df = pd.DataFrame(
        {
            "MP2": std_dev * np.random.randn(n_samples) + mean,
            "HF": std_dev * np.random.randn(n_samples) - mean,
            "MP2.5": std_dev * np.random.randn(n_samples) + mean,
        }
    )
    # Only specify columns you want to plot
    vals = {
        "MP2 label": "MP2",
        "HF label": "HF",
    }
    violin_plot(df, vals, ylim=[-20, 35], output_filename="example")
