from os import makedirs
import pandas as pd
import torchmetrics
import torch

from src import figures


def get_testcase_rankingmetrics(test_case: pd.DataFrame):

    # Input: model probabilities and targets of a test case
    # Output: percentile of this test case and raw position of the author's image

    sorted_ranking = test_case.sort_values(
        'pred', ascending=False).reset_index(drop=True)

    dev_position = sorted_ranking['is_dev'].idxmax()

    return pd.DataFrame({'dev_position': [dev_position],
                         'percentile': [dev_position/len(sorted_ranking)],
                         'id_user': sorted_ranking['id_user'][0],
                         'id_restaurant': sorted_ranking['id_restaurant'][0]})


def test_tripadvisor_authorship_task(datamodule, model_preds):

    makedirs('docs/'+datamodule.city, exist_ok=True)

    # Data for the percentile figures
    percentile_figure_data = {'city': datamodule.city, 'metrics': []}
    recall_figure_data = {'city': datamodule.city, 'metrics': []}
    ndcg_figure_data = {'city': datamodule.city, 'metrics': []}

    for model in model_preds:
        print("="*50)
        print(model)
        print("="*50)

        test_set = datamodule.test_dataset.dataframe
        test_set['pred'] = model_preds[model]

        train_set = datamodule.train_dataset.dataframe

        # Get no. of images in each test case: not the same unique restaurant
        # images, as a user may have >1 images per restaurant and each test case
        # only has one of them
        images_per_testcase = test_set.value_counts(
            'id_test').reset_index().rename(columns={0: 'testcase_num_images'})

        # Compute number of photos in each user's train set
        train_photos_per_user = train_set[train_set['take'] == 1].drop_duplicates(
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
        model_percentile_metrics = {'min_photos': [],
                                    'num_test_cases': [],
                                    'median_percentile': [],
                                    'model_name': model
                                    }

        # We only take into account restaurants with >10 photos
        test_cases = test_cases[test_cases['testcase_num_images'] >= 10]

        # Compute percentile figure metrics
        print(f"Min. imgs  Percentile  Test Cases")
        for i in range(1, 101):
            percentiles = test_cases[test_cases['author_num_train_photos']
                                     >= i]['percentile']

            model_percentile_metrics['min_photos'].append(i)
            model_percentile_metrics['num_test_cases'].append(len(percentiles))
            model_percentile_metrics['median_percentile'].append(
                percentiles.median())

            print(f"{i:<11}{percentiles.median():<12.3f}({len(percentiles)})")
        percentile_figure_data['metrics'].append(model_percentile_metrics)

        # For the recall metric, only include users with >= train images
        test_cases = test_cases[test_cases['author_num_train_photos'] >= 10]

        # Initialize recall table data
        model_recall_metrics = {'k': [], 'Recall': [], 'model_name': model}
        model_ndcg_metrics = {'k': [], 'NDCG': [], 'model_name': model}

        test_set = test_set[test_set['id_test'].isin(
            test_cases['id_test'])].reset_index(drop=True)

        preds = torch.tensor(test_set['pred'], dtype=torch.float)
        target = torch.tensor(test_set['is_dev'], dtype=torch.long)
        indexes = torch.tensor(test_set['id_test'], dtype=torch.long)
        # % of test cases where the image was in position k=1,2,3...10 (Recall at k)
        print("k  Recall  NDCG")
        for k in range(1, 10+1):
            recall_k = torchmetrics.RetrievalRecall(k=k)(
                preds=preds, target=target, indexes=indexes
            )
            model_recall_metrics['k'].append(k)
            model_recall_metrics['Recall'].append(recall_k)

            ndcg_k = torchmetrics.RetrievalNormalizedDCG(k=k)(
                preds=preds, target=target, indexes=indexes
            )

            model_ndcg_metrics['k'].append(k)
            model_ndcg_metrics['NDCG'].append(ndcg_k)
            print(f"{k:<3}{recall_k:<8.3f}{ndcg_k:.3f}")

        recall_figure_data['metrics'].append(model_recall_metrics)
        ndcg_figure_data['metrics'].append(model_ndcg_metrics)

    figures.retrieval_figure(recall_figure_data, 'Recall')
    figures.retrieval_figure(ndcg_figure_data, 'NDCG')
    figures.percentile_figure(percentile_figure_data)
