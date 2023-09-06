import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

PLOT_COLORS = {
    "ELVis": "g",
    "PRESLEY": "r",
    "MF_ELVis": "b",
    "RANDOM": "y",
    "CNT": "purple",
}


def percentile_figure(data: dict):
    rcParams["figure.figsize"] = 9, 5
    # Font size
    plt.rcParams.update({"font.size": 17})

    # Left plot : percentile metric
    for metrics in data["metrics"]:
        plt.plot(
            metrics["min_photos"],
            metrics["median_percentile"],
            linewidth=3.0,
            label=metrics["model_name"]
            if metrics["model_name"] != "PRESLEY"
            else "BRIE",
            alpha=0.8,
            color=PLOT_COLORS[metrics["model_name"]],
        )

    # Unicode character for geq

    plt.xlabel("Users with \u2265x train images")
    plt.ylabel("Median percentile\nof author's image")

    plt.xlim(0, 100)
    plt.ylim(0, 1)

    # plt.xticks([1] + list(range(5, 101, 5)))
    # plt.yticks(np.arange(0, 1 + 0.1, 0.1))

    plt.grid(True, linestyle="--", color="lightgray")
    if data["city"] == "gijon":
        plt.legend(loc="upper left")
    plt.tight_layout()

    # Right plot: test cases
    plt.twinx()
    plt.plot(
        metrics["min_photos"],
        metrics["num_test_cases"],
        linewidth=3.0,
        color="black",
        label="Test cases",
        alpha=0.3,
    )

    plt.ylabel("Test cases")
    plt.yscale("log")

    if data["city"] == "gijon":
        plt.legend(loc="upper right")

    plt.title(data["city"])
    # Output
    plt.savefig(
        f'figures/{data["city"]}/percentile_{data["city"]}.pdf', bbox_inches="tight"
    )
    # plt.show()


def retrieval_figure(data: dict, metric_name: str):
    rcParams["figure.figsize"] = 4, 4

    plt.title(metric_name)

    for metrics in data["metrics"]:
        plt.plot(
            metrics["k"],
            metrics[metric_name],
            linewidth=2.0,
            label=metrics["model_name"],
        )

    plt.xlabel("Position k of ranking")
    plt.ylabel(f"{metric_name} at k")

    plt.xlim(1, 10)

    # Make the lim a bit larger than the max value
    plt.ylim(0, max([max(metrics[metric_name]) for metrics in data["metrics"]]) * 1.1)

    plt.xticks(range(1, 11, 1))

    plt.grid(True, linestyle="--", color="lightgray")
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Output
    plt.savefig(f'docs/{data["city"]}/{metric_name}.pdf', bbox_inches="tight")
    plt.show()
