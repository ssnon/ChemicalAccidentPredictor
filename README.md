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
#### We used the chemical accident dataset provided by the National Institute of Chemical Safety(NICS).
#### link : https://www.data.go.kr/data/15069200/fileData.do#
#### Each train, val, test data should be located in dataset/train/train.csv, dataset/valid/valid.csv and dataset/test/test.csv respectively.

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

#### to use augmentation, use
- data_augmentation.py --data_directory="your_data_directory_path" --input_file_name"your_input_file_name.csv" --output_file_name="your_output_file_name.csv" --aug_ratio=your_ratio --cossim_threshold=your_threshold
- you can see arguments in data_augmentation.py

## Networks
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
- [Kobert](https://github.com/SKTBrain/KoBERT)