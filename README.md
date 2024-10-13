# Text classification

## Prerequests
> Python3.12 is used

1. Install requirements.txt
```bash
pip3 install -r requirements.txt
```

## EDA
Dataset is taken from [kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data), navigate to more description. To brief, dataset consist of two columns with `statement` and `status` name. The `status` is our target, which could take seven different value, so we deal with multiclass classification. The total amount of row is 52681, some of this rows are nan, so we remove them. It lead to 53043 appropriate rows. Below you could see the distribution of our targets(or classes).
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
 