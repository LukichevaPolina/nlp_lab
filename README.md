# Text classification

## Prerequests
> Python3.12 is used

1. Clone the reposiroty
```bash
git clone https://github.com/LukichevaPolina/nlp_lab.git
cd nlp_lab
```

2. Install requirements.txt
```bash
pip3 install -r requirements.txt
```

3. Set up `PYTHONPATH`
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
```

## EDA
Dataset is taken from [kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data), navigate to more description. To brief, dataset consist of two columns with `statement` and `status` name. The `status` is our target, which could take seven different value, so we deal with multiclass classification. The total amount of row is 53043, some of this rows are nan, so we remove them. It lead to 52681 appropriate rows. Below you could see the distribution of our targets(or classes).
![alt text](./graphs/class_distribution.png)

As you can see we have disbalanced classes, we deal with it using [class weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class-weight.compute_class-weight.html) approach. Below we provide a little more statistics of our data.

![alt text](./graphs/class_features_distribution_length.png)
> The chart consist of histograms of the sentences length. Notice that a huge frequency corresponds to relative short sentences. A long sentences relate to Suicidal, Personality disorder, following by Anxiety, Depression, Bipolar and Stress. A smallest sentences belong to Normal.

![alt text](./graphs/class_features_distribution_punctuation_length.png)
> The chart consist of histograms of the amount of punctuation (or punctuation length). Notice that the sentences include only one punctuation frequently.

![alt text](./graphs/class_features_distribution_digit_length.png)
> The chart consist of histograms of the digit amount. Notice that more sentence have few digits.

We conclude that our data include short sentences around 5-50 words for `Normal`, around 1000-1500 of that for `Suicidal` and `Personality disorder`, about 500-600 of that for `Anxiety`, `Depression`, `Bipolar` and `Stress`, few punctuation and few digits. So the statistics say that `Suicidal` and `Personality disorder` people have a numerous thoughts, probably unnecessary. Notice also that almost all histograms are left shifted. 

Below we provide the histograms of more frequent words in our data.

![alt text](./graphs/data_info.png)
> The chart consist of histogram of the top 20 frequently words. The top 3 words are `feel`, `like`, `want`.

One of our preprocess include following steps: `drop_nan` -> `remove_punctuation` -> `remove_digits` -> `remove_stop_words` -> `tokenize` -> `stemming` -> `lemmatization`. The embeddings strategy is `tfidf`. We want to exam the four models: `svm`, `decision_tree`, `cnn` and `linear`.

## Classical algorithms
### SVM
**To train**
```bash
python3 main.py --dataset_path {dataset_path} --algorithm svm --embeddigns tfidf --class_balancer class-weight --preprocessor remove-all --mode train
```

**To eval**
```bash
python3 main.py --dataset_path {dataset_path} --algorithm svm --embeddigns tfidf --class_balancer class-weight --preprocessor remove-all --mode eval
```

#### Val metrics
| class | f1_score |
| ----- | -------- |
| 0     | 0.78     |
| 1     | 0.77     |
| 2     | 0.69     |
| 3     | 0.90     |
| 4     | 0.64     |
| 5     | 0.54     |
| 6     | 0.65     |

**accuracy** = 0.73, **f1_weighted** = 0.75

##### Details
Grid parametrs search was made. Best parameters from gridsearch: {'C': 1, 'multi_class': 'ovr'}

### Decision Tree

**To train**
```bash
python3 main.py --dataset_path {dataset_path} --algorithm decision-tree --embeddigns tfidf --class_balancer class-weight --preprocessor remove-all --mode train
```

**To eval**
```bash
python3 main.py --dataset_path {dataset_path} --algorithm decision-tree --embeddigns tfidf --class_balancer class-weight --preprocessor remove-all --mode eval
```

#### Val metrics
| class | f1_score |
| ----- | -------- |
| 0     | 0.59     |
| 1     | 0.58     |
| 2     | 0.58     |
| 3     | 0.78     |
| 4     | 0.50     |
| 5     | 0.32     |
| 6     | 0.52     |

**accuracy** = 0.57, **f1_weighted** = 0.62

##### Details
Grid parametrs search was made. Best parameters from gridsearch: {'criterion': 'gini', 'max_depth': 12}

## DL algorithms
### CNN
**To train**
```bash
python3 main.py --dataset_path {dataset_path} --algorithm cnn --embeddigns tfidf --class_balancer class-weight --preprocessor remove-all --mode train
```

**To eval**
```bash
python3 main.py --dataset_path {dataset_path} --algorithm cnn --embeddigns tfidf --class_balancer class-weight --preprocessor remove-all --mode eval
```

#### Train metrics

| Learning Curve | Accuracy Curve | F1 Curve |
:---------------:|:--------------:|:---------:
![alt text](./graphs/cnn_learning_curve.png) | ![alt text](./graphs/cnn_accuracy_curve.png) | ![alt text](./graphs/cnn_f1_curve.png)

#### Val metrics
| class | f1_score |
| ----- | -------- |
| 0     | 0.76     |
| 1     | 0.78     |
| 2     | 0.69     |
| 3     | 0.88     |
| 4     | 0.69     |
| 5     | 0.56     |
| 6     | 0.64     |

**accuracy** = 0.71, **f1_weighted** = 0.74

### Linear
**To train**
```bash
python3 main.py --dataset_path {dataset_path} --algorithm linear --embeddigns tfidf --class_balancer class-weight --preprocessor remove-all --mode train
```

**To eval**
```bash
python3 main.py --dataset_path {dataset_path} --algorithm linear --embeddigns tfidf --class_balancer class-weight --preprocessor remove-all --mode eval
```

#### Train metrics
| Learning Curve | Accuracy Curve | F1 Curve |
:---------------:|:--------------:|:---------:
![alt text](./graphs/linear_learning_curve.png) | ![alt text](./graphs/linear_accuracy_curve.png) | ![alt text](./graphs/linear_f1_curve.png)

#### Val metrics
| class | f1_score |
| ----- | -------- |
| 0     | 0.63     |
| 1     | 0.56     |
| 2     | 0.61     |
| 3     | 0.84     |
| 4     | 0.49     |
| 5     | 0.42     |
| 6     | 0.59     |

**accuracy** = 0.61, **f1_weighted** = 0.67

## Total Comparing
The best model 
![alt text](./graphs/plot_metrics_all.png)


## Experiements
### Prepocessing 
We experemented with preprocessing on classical models. We have three types of experiements: 
1. Preprocessing (removing all punctuation, digits and stop-words) + lemmatization
2. Preprocessing (removing all punctuation) + lemmatization
3. Preprocessing (removing all punctuation, digits and stop-words) + stemming.

As evidenced by the plots, there is no significant difference between the results. This is likely due to the fact that the models focused on the "specific" aspect of the person condition.

| SVM | Decisin Tree |
:----:|:--------------:
![alt text](./graphs/plot_metrics_svm.png) | ![alt text](./graphs/plot_metrics_decision_tree.png)

### Parametr searching
#### Classical models
Grid search was aplied. For SVC models were checked {"C": [1, 10, 100, 1000], "multi_class": ["ovr", "crammer_singer"]} parametrs.
For Decision Tree were checked {'criterion': [gini', 'entropy'], 'max_depth': np.arange(3, 15)} parametrs.
#### DL models
A series of experiments was conducted to test the impact of varying the learning rate (1e-3, 2e-5, 2e-3) at the optimizer, regularization weight, and scheduler. Additionally, the batch size was modified from 64 to 128. To account for class imbalance, different weights were used in cross entropy.
