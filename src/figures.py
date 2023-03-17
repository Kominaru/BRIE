import matplotlib.pyplot as plt
from matplotlib import rcParams

def percentile_figure(data: dict) -> None:

    rcParams['figure.figsize'] = 10,7
    plt.plot(data['min_photos'],data['median_percentile'])
    plt.xlabel('Users with >=x train images')
    plt.ylabel('Median ranking percentile\nof author\'s image')
    plt.savefig('docs/barcelona/percentile.pdf')
    plt.show()