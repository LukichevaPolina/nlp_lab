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