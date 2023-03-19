import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


def percentile_figure(data: dict) -> None:

    rcParams['figure.figsize'] = 8, 4

    plt.plot(data['min_photos'], data['median_percentile'],
             linewidth=3.0, color='green', label=data['model_name'])

    plt.xlabel('Users with >=x train images')
    plt.ylabel('Median ranking percentile\nof author\'s image')

    plt.xlim(0, 100)
    plt.ylim(0, 1)

    plt.xticks([1]+list(range(5, 101, 5)))
    plt.yticks(np.arange(0, 1+0.1, .1))

    plt.grid(True, linestyle='--', color="lightgray")

    plt.legend(loc='upper left')

    plt.twinx()
    plt.plot(data['min_photos'], data['num_test_cases'],
             linewidth=3.0, color='black', label='Test cases', alpha=0.3)

    plt.ylabel('Available test cases')
    plt.yscale('log')

    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(f'docs/{data["city"]}/percentile.pdf')
    plt.show()
