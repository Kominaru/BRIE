from os import makedirs
import pandas as pd
import torchmetrics
import torch

from src.figures import percentile_figure


def test_tripadvisor_authorship_task(dataset, predictions, model_name):

    makedirs('docs/'+dataset.city, exist_ok=True)

    test_set = dataset.test_data.samples
    test_set['pred'] = predictions
    print(torchmetrics.functional.accuracy(torch.Tensor(
        test_set['pred']), torch.Tensor(test_set['is_dev']), task='binary'))
    print(torchmetrics.functional.recall(torch.Tensor(
        test_set['pred']), torch.Tensor(test_set['is_dev']), task='binary'))
    print(torchmetrics.functional.specificity(torch.Tensor(
        test_set['pred']), torch.Tensor(test_set['is_dev']), task='binary'))

    def get_testsample_rankingmetrics(test_case: pd.DataFrame):
        test_case = test_case.sort_values(
            'pred', ascending=False).reset_index(drop=True)
        dev_position = test_case['is_dev'].idxmax()
        return dev_position/len(test_case), dev_position

    test_cases = test_set[test_set['is_dev'] == 1].reset_index(drop=True)
    train_photos_per_user = dataset.train_data.samples[dataset.train_data.samples['take'] == 1].drop_duplicates(
        keep='first').value_counts('id_user').reset_index().rename(columns={0: 'author_num_train_photos'})
    test_photos_per_restaurant = dataset.test_data.samples.drop_duplicates(['id_restaurant', 'id_img'], keep='first').value_counts(
        'id_restaurant').reset_index().rename(columns={0: 'rest_num_test_photos'})

    percentiles = test_set.groupby('id_test').apply(
        get_testsample_rankingmetrics).reset_index().rename(columns={0: 'ranking_metrics'})
    print(percentiles)
    percentiles[['percentile', 'dev_position']] = pd.DataFrame(
        percentiles['ranking_metrics'].tolist(), index=percentiles.index)
    test_cases = pd.merge(test_cases, percentiles,
                          left_on='id_test', right_on='id_test', how='inner')
    test_cases = pd.merge(test_cases, train_photos_per_user,
                          left_on='id_user', right_on='id_user', how='inner')
    test_cases = pd.merge(test_cases, test_photos_per_restaurant,
                          left_on='id_restaurant', right_on='id_restaurant', how='inner')

    test_cases = test_cases[test_cases['rest_num_test_photos'] > 10]

    percentile_figure_data = {'min_photos': [],
                              'num_test_cases': [],
                              'median_percentile': [],
                              'city': dataset.city,
                              'model': model_name}
    for i in range(1, 100):
        percentiles = test_cases[test_cases['author_num_train_photos']
                                 >= i]['percentile']

        print(f"{i}\t{percentiles.median():.3f}\t({len(percentiles)})")

        percentile_figure_data['min_photos'].append(i)
        percentile_figure_data['num_test_cases'].append(len(percentiles))
        percentile_figure_data['median_percentile'].append(
            percentiles.median())

    test_cases = test_cases[test_cases['author_num_train_photos'] > 10]

    percentile_figure(percentile_figure_data)

    recall_figure_data = {'k': [], 'recall': []}
    for i in range(10):
        top_i = len(test_cases[(test_cases['dev_position']) <= i])

        print(f"TOP-{i+1}\t{top_i/len(test_cases):.3f}")

        recall_figure_data['k'].append(i)
        recall_figure_data['recall'].append(top_i/len(test_cases))
