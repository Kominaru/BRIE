from os import makedirs
import pandas as pd
import torchmetrics
import torch

from src.figures import percentile_figure
import numpy as np

# Input: model probabilities and targets of a test case
# Output: percentile of this test case and raw position of the author's image


def get_testcase_rankingmetrics(test_case: pd.DataFrame):
    sorted_ranking = test_case.sort_values(
        'pred', ascending=False).reset_index(drop=True)

    dev_position = sorted_ranking['is_dev'].idxmax()

    return pd.DataFrame({'dev_position': [dev_position],
                         'percentile': [dev_position/len(sorted_ranking)],
                         'id_user': sorted_ranking['id_user'][0],
                         'id_restaurant': sorted_ranking['id_restaurant'][0]})


def test_tripadvisor_authorship_task(dataset, predictions, model_name):

    makedirs('docs/'+dataset.city, exist_ok=True)

    test_set = dataset.test_data.samples
    test_set['pred'] = predictions

    # Compute number of photos in each test case ranking
    # May not be the same as the number of unique photos in each restaurant,
    # as users sometimes have up to 4 images in the same restaurant
    # and each of them only appears in one test case
    images_per_testcase = test_set.value_counts(
        'id_test').reset_index().rename(columns={0: 'testcase_num_images'})

    # Compute number of photos in each user's train set
    train_photos_per_user = dataset.train_val_data.samples[dataset.train_val_data.samples['take'] == 1].drop_duplicates(
        keep='first').value_counts('id_user').reset_index().rename(columns={0: 'author_num_train_photos'})

    # # Compute the percentile metric of each test case
    test_cases = test_set.groupby('id_test').apply(
        get_testcase_rankingmetrics).reset_index()

    # Add the user and subreddit information
    test_cases = pd.merge(test_cases, train_photos_per_user,
                          left_on='id_user', right_on='id_user', how='inner')
    test_cases = pd.merge(test_cases, images_per_testcase,
                          left_on='id_test', right_on='id_test', how='inner')

    # Initialize figure data
    percentile_figure_data = {'min_photos': [],
                              'num_test_cases': [],
                              'median_percentile': [],
                              'city': dataset.city,
                              'model_name': model_name}

    # We only take into account restaurants with >10 photos
    test_cases = test_cases[test_cases['testcase_num_images'] >= 10]

    # Compute percentile figure metrics
    print(f"Min. imgs  Percentile  Test Cases")
    for i in range(1, 101):
        percentiles = test_cases[test_cases['author_num_train_photos']
                                 >= i]['percentile']

        percentile_figure_data['min_photos'].append(i)
        percentile_figure_data['num_test_cases'].append(len(percentiles))
        percentile_figure_data['median_percentile'].append(
            percentiles.median())

        print(f"{i:<11}{percentiles.median():<12.3f}({len(percentiles)})")

    percentile_figure(percentile_figure_data)

    # For the recall metric, only include users with >= train images
    test_cases = test_cases[test_cases['author_num_train_photos'] >= 10]

    # Initialize recall table data
    recall_table_data = {'k': [], 'recall': []}

    # Test cases where the image was in position 1,2,3...10
    for i in range(10):
        top_i = len(test_cases[(test_cases['dev_position']) <= i])

        recall_table_data['k'].append(i+1)
        recall_table_data['recall'].append(top_i/len(test_cases))

        print(f"TOP-{i+1}\t{top_i/len(test_cases):.3f}")
