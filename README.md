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

As you can see we have disbalanced classes, we deal with it using [class weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) approach. Below we provide a little more statistics of our data.

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

One of our preprocess include following steps: `drop_nan` -> `remove_punctuation` -> `remove_digits` -> `remove_stop_words` -> `tokenize` -> `stemming` -> `lemmatization`. The embeddings strategy is `tfidf`. We want to exam the four models: `svm`, `decision_tree`, `cnn` and `lstm`.

## Classical algorithms
### SVM
### Decision Tree
### Comparing
Here need to add graph with metrics for two models

## DL algorithms
### CNN
```bash
python3 main.py --dataset_path {dataset_path} --algorithm cnn --embeddigns tfidf --class_balancer class_weight --preprocessor remove_all --mode eval
```
Here need to add graph with metrics for one model
Here need to add graph woth train/val losses for one model

### LSTM
### Comparing
Here need to add graph with metrics for two models

Here need to add graph with losses for two models

## Total Comparing
Here need to add graph with metrics for four models

## Example of running
python3 main.py --dataset_path {dataset_path} --algorithm svm --embeddigns tfidf --class_balancer class_weight --preprocessor remove_all --mode eval