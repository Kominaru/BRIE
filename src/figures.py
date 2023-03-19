import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


def percentile_figure(data: dict):

    rcParams['figure.figsize'] = 8, 4

    # Left plot : percentile metric
    for metrics in data['metrics']:

        plt.plot(metrics['min_photos'], metrics['median_percentile'],
                 linewidth=3.0, label=metrics['model_name'])

    plt.xlabel('Users with >=x train images')
    plt.ylabel('Median ranking percentile\nof author\'s image')

    plt.xlim(0, 100)
    plt.ylim(0, 1)

    plt.xticks([1]+list(range(5, 101, 5)))
    plt.yticks(np.arange(0, 1+0.1, .1))

    plt.grid(True, linestyle='--', color="lightgray")
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Right plot: test cases
    plt.twinx()
    plt.plot(metrics['min_photos'], metrics['num_test_cases'],
             linewidth=3.0, color='black', label='Test cases', alpha=0.3)

    plt.ylabel('Available test cases')
    plt.yscale('log')

    plt.legend(loc='upper right')

    # Output
    plt.savefig(f'docs/{data["city"]}/percentile.pdf', bbox_inches='tight')
    plt.show()


def recall_figure(data: dict):

    rcParams['figure.figsize'] = 4, 4

    for metrics in data['metrics']:

        plt.plot(metrics['k'], metrics['recall'],
                 linewidth=3.0, label=metrics['model_name'])

    plt.xlabel('Position k of ranking')
    plt.ylabel('Recall at k')

    plt.xlim(1, 10)
    plt.ylim(0, 1)

    plt.xticks(range(1, 11, 1))
    plt.yticks(np.arange(0, 1+0.1, .1))

    plt.grid(True, linestyle='--', color="lightgray")
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Output
    plt.savefig(f'docs/{data["city"]}/recall.pdf', bbox_inches='tight')
    plt.show()
