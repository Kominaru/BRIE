# <div align="center"> BRIE : Bayesian Ranking of Images for explainability </div>

### <div align="center"> Jorge Paz-Ruza*, Amparo Alonso-Betanzos <br> Berta Guijarro-Berdiñas, Brais Cancela, Carlos Eiras-Franco <br> <br> [Sustainable Transparency in Recommender Systems: <br> Bayesian Ranking of Images for Explainability](https://doi.org/10.1016/j.inffus.2024.102497) </div>

##### <div align="center"> Published on <b>Information Fusion</b>, Vol. 108, 2024</div>







<br>

<div align="center"><img src="https://media.licdn.com/dms/image/v2/D5622AQHXcMV1koKxrw/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1717443614322?e=1730332800&v=beta&t=2F_jRrOCeF55B4Qv1ei_Q-rewvdL7PYJ1Jm1KIsqDvI" width="400"></div>

<br>

<div align="center">
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sustainable-transparency-in-recommender/image-based-recommendation-explainability-on"/>
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sustainable-transparency-in-recommender/image-based-recommendation-explainability-on-5"/>
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sustainable-transparency-in-recommender/image-based-recommendation-explainability-on-2"/>
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sustainable-transparency-in-recommender/image-based-recommendation-explainability-on-3"/>
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sustainable-transparency-in-recommender/image-based-recommendation-explainability-on-4"/>
    <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sustainable-transparency-in-recommender/image-based-recommendation-explainability-on-1"/>
</div>


### 1. Abstract

<p align="justify"> Recommender Systems have become crucial in the modern world, commonly guiding users towards relevant content or products, and having a large influence over the decisions of users and citizens. However, ensuring transparency and user trust in these systems remains a challenge; personalized explanations have emerged as a solution, offering justifications for recommendations. Among the existing approaches for generating personalized explanations, using existing visual content created by users is a promising option to maximize transparency and user trust. State-of-the-art models that follow this approach, despite leveraging highly optimized architectures, employ surrogate learning tasks that do not efficiently model the objective of ranking images as explanations for a given recommendation; this leads to a suboptimal training process with high computational costs that may not be reduced without affecting model performance. This work presents BRIE, a novel model where we leverage Bayesian Pairwise Ranking to enhance the training process, allowing us to consistently outperform state-of-the-art models in six real-world datasets while reducing its model size by up to 64 times and its CO
 emissions by up to 75% in training and inference.</p>

### 2. Setup

#### 2.1. Environment
- The code in this repository has been tested with Python 3.10 and Cuda 11.8
- You can install all required packages with `pip install -r requirements.txt`
- This framework was executed in a dedicated Windows 10 Pro machine with an Intel Core i7-10700K CPU @ 3.80GHz, 16GB RAM, and an NVIDIA GeForce 2060 GPU Super.

#### 2.2. Datasets
- The six datasets are available for download at https://zenodo.org/record/5644892
- datasets should be placed in the `data` folder, conforming the following structure:
    ```
    data
    ├── barcelona
    │   ├── data_10+10
    │   │   ├── IMG_TRAIN
    │   │   ├── IMG_TEST
    │   │   ├── ...
    │   ├── original_take
    │   │   ├── ...
    │   
    ├── madrid
    │   ├── ...
    │
    ├── ...
    ```

#### 2.3. Pre-trained BRIE models
- Pre-trained BRIE models are available for download at https://drive.google.com/drive/folders/1y5HlMk3tyQyW2nNEeaRBR5rpnVJwh5K3
- Models should be placed in the `models` folder, conforming the following structure:
    ```
    models
    ├── barcelona
    │   ├── BRIE
    │   │   ├── best_model.ckpt
    │   
    ├── madrid
    │   ├── ...
    │
    ├── ...
    ```
### 3. Usage

#### 3.1. Training
- To train a BRIE model, run `python main.py --stage train --city CITY_NAME --model BRIE --max_epochs EPOCHS [--batch_size BATCH_SIZE] [--lr LR] [--dropout DROPOUT] -d DIMS --workers NUM_WORKERS [--early_stopping] [--no_validation]`
- For example, to train BRIE with the hyperparameters used in the paper, run `python main.py --city barcelona --model BRIE --max_epochs 15 --batch_size 2**14 --lr 0.001 --dropout 0.75 -d 64 --workers 4 --no_validation`

#### 3.2. Evaluation
- To test a BRIE model, run `python main.py --stage test --city CITY_NAME --model MODEL_NAME... [--batch_size BATCH_SIZE] --workers NUM_WORKERS`
- Multiple models can be tested at once by specifying their names separated by spaces after the `--model` argument
- For example, to test BRIE against ELVis, run `python main.py --stage test --city barcelona --model BRIE ELVis --batch_size 2**14 --workers 4`

### 4. Results

- Below are the performance results obtained by BRIE in the six datasets used in the paper, compared to the state-of-the-art models ELVis and MF-ELVis, as well as two basic baselines. 

<div align="center">

|          | **Gijón**   |           |           |   | **Barcelona** |           |           |   | **Madrid** |           |           |
|----------|-------------|----------:|----------:|---|---------------|----------:|----------:|---|------------|----------:|----------:|
|          | MRecall@10  | MNDCG@10  | MAUC      |   | MRecall@10    | MNDCG@10  | MAUC      |   | MRecall@10 | MNDCG@10  | MAUC      |
| RND      |       0.373 |     0.185 |     0.487 |   |         0.409 |     0.186 |     0.502 |   |      0.374 |     0.171 |     0.499 |
| CNT      |       0.464 |     0.218 |     0.546 |   |         0.443 |     0.219 |     0.554 |   |      0.420 |     0.203 |     0.557 |
| ELVis    |       0.521 |     0.262 |     0.596 |   |         0.597 |     0.327 |     0.631 |   |      0.572 |     0.314 |     0.638 |
| MF-ELVis |       0.538 |     0.285 |     0.592 |   |         0.557 |     0.293 |     0.596 |   |      0.528 |     0.279 |     0.601 |
| BRIE     |   **0.607** | **0.333** | **0.643** |   |     **0.630** | **0.368** | **0.663** |   |  **0.612** | **0.348** | **0.673** |
|          |             |           |           |   |               |           |           |   |            |           |           |
|          | **Newyork** |           |           |   | **Paris**     |           |           |   | **London** |           |           |
|          | MRecall@10  | MNDCG@10  | MAUC      |   | MRecall@10    | MNDCG@10  | MAUC      |   | MRecall@10 | MNDCG@10  | MAUC      |
| RND      |       0.374 |     0.168 |     0.502 |   |         0.459 |     0.209 |     0.502 |   |      0.342 |     0.155 |     0.500 |
| CNT      |       0.431 |     0.217 |     0.563 |   |         0.499 |     0.245 |     0.557 |   |      0.400 |     0.200 |     0.562 |
| ELVis    |       0.553 |     0.304 |     0.637 |   |         0.643 |     0.352 |     0.630 |   |      0.530 |     0.293 |     0.629 |
| MF-ELVis |       0.516 |     0.276 |     0.602 |   |         0.606 |     0.323 |     0.596 |   |      0.531 |     0.267 |     0.597 |
| BRIE     |   **0.598** | **0.341** | **0.677** |   |     **0.669** | **0.391** | **0.666** |   |  **0.563** | **0.318** | **0.665** |

</div>




- The results for ELVis and MF-ELVis were obtained by running the code provided by the authors of the original papers: [MF-ELVis](https://github.com/Kominaru/tfg-komi), [ELVis](https://github.com/pablo-pnunez/ELVis)

- Sustainability comparisons (training time and emissions, model size, and inference time and emissions) can be found in the paper.

### 5. Citation

- If you use this code or reference this model, we encourage citing the journal paper:

  - APA:
    ```
    Paz-Ruza, J., Alonso-Betanzos, A., Guijarro-Berdiñas, B., Cancela, B., & Eiras-Franco, C. (2024). Sustainable transparency on recommender systems: Bayesian ranking of images for explainability. Information Fusion, 102497.
    ```

  - Bibtex:
    ```
    @article{paz2024sustainable,
    title={Sustainable transparency on recommender systems: Bayesian ranking of images for explainability},
    author={Paz-Ruza, Jorge and Alonso-Betanzos, Amparo and Guijarro-Berdi{\~n}as, Bertha and Cancela, Brais and Eiras-Franco, Carlos},
    journal={Information Fusion},
    pages={102497},
    year={2024},
    publisher={Elsevier}
    }
    ```
