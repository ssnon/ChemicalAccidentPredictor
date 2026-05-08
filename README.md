## Experiment environment
#### cpu : Intel(R) Xeon(R) Platinum 8158 CPU @ 3.00GHz
#### gpu : NVIDIA GeForce RTX 2080 Ti
#### os : Ubuntu 20.04.6 LTS 

## Requirements
#### python : 3.10.20
#### CUDA : 12.8

## Environment
#### pip install -r requirements.txt

## Dataset
#### we use chemical accident dataset which provided by National Institute of Chemical Safety(NICS).
#### link : https://www.data.go.kr/data/15069200/fileData.do#
#### each train, val, test data should be located in dataset/train/train.csv, dataset/valid/valid.csv and dataset/test/test.csv repectively.

## command
#### to train model, use
- ./run_experiment.sh
#### if you want to modify arguments, use like this
- main.py --data_directory="your_directory" --model="model_name"
- you can see more arguments in main.py

#### to analysis model, use
- ./run_analysis.sh
#### if you want to modify arguments, use like this
- lime_analysis_kobert.py --data_directory="your_directory" --model="model_name"
- lime_analysis_classicML.py --data_directory="your_directory" --model="model_name"
- you can see more arguments in lime_analysis_kobert.py and lime_analysis_classicML.py

## Networks
- Logistic Regression ![](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Random Forest ![](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- SVM ![](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- XGBoost ![](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
- Kobert ![](https://github.com/SKTBrain/KoBERT)