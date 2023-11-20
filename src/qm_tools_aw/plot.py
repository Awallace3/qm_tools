import os
# import seaborn as sns
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
    vals: {},
    title_name: str,
    pfn: str,
    bottom: float = 0.4,
    ylim=[-15, 35],
    transparent=True,
    widths=0.85,
    figure_size=None,
    set_xlable=False,
    x_label_rotation=90,
    x_label_fontsize=8,
    ylabel=r"Error ($\mathrm{kcal\cdot mol^{-1}}$)",
    dpi=1200,
    pdf=False,
    usetex=True,

) -> None:
    """
    Plot a violin plot of the data in df. For vals,
    specify {"output_name": "column_name",}.
    """
    print(f"Plotting {pfn}")
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []

    annotations = []  # [(x, y, text), ...]
    cnt = 1
    plt.rcParams["text.usetex"] = usetex
    for k, v in vals.items():
        df[v] = pd.to_numeric(df[v])
        df_sub = df[df[v].notna()].copy()
        vData.append(df_sub[v].to_list())
        k_label = "\\textbf{" + k + "}"
        vLabels.append(k_label)
        m = df_sub[v].max()
        rmse = df_sub[v].apply(lambda x: x**2).mean() ** 0.5
        mae = df_sub[v].apply(lambda x: abs(x)).mean()
        max_error = df_sub[v].apply(lambda x: abs(x)).max()
        text = r"$\mathit{%.2f}$" % mae
        text += "\n"
        text += r"$\mathbf{%.2f}$" % rmse
        text += "\n"
        text += r"$\mathrm{%.2f}$" % max_error
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
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
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
    # TODO: fix minor ticks to be between
    navy_blue = (0.0, 0.32, 0.96)
    ax.set_xticks(xs)
    # minor_yticks = np.arange(ylim[0], ylim[1], 2)
    # ax.set_yticks(minor_yticks, minor=True)

    plt.setp(ax.set_xticklabels(vLabels), rotation=x_label_rotation, fontsize=x_label_fontsize)
    ax.set_xlim((0, len(vLabels)))
    if ylim is not None:
        ax.set_ylim(ylim)

        minor_yticks = create_minor_y_ticks(ylim)
        ax.set_yticks(minor_yticks, minor=True)

    lg = ax.legend(loc="upper left", edgecolor="black", fontsize="8")
    # lg.get_frame().set_alpha(None)
    # lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    if set_xlable:
        ax.set_xlabel("Level of Theory", color="k")
    ax.set_ylabel(ylabel, color="k")
    # ax.grid(color="gray", which="major", linewidth=0.5, alpha=0.3)
    # ax.grid(color="gray", which="minor", linewidth=0.5, alpha=0.3)

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

    if title_name is not None:
        plt.title(f"{title_name}")
    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=bottom)

    if pdf:
        fn_pdf = f"plots/{pfn}_dbs_violin.pdf"
        fn_png = f"plots/{pfn}_dbs_violin.png"
        plt.savefig(
            fn_pdf, transparent=transparent, bbox_inches="tight", dpi=dpi,
        )
        if os.path.exists(fn_png):
            os.system(f"rm {fn_png}")
        os.system(f"pdftoppm -png -r 400 {fn_pdf} {fn_png}")
        if os.path.exists(f"{fn_png}-1.png"):
            os.system(f"mv {fn_png}-1.png {fn_png}")
        else:
            print(f"Error: {fn_png}-1.png does not exist")
    else:
        plt.savefig(
            f"plots/{pfn}_dbs_violin.png", transparent=transparent, bbox_inches="tight", dpi=dpi,
        )
    plt.clf()
    return
