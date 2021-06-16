from __future__ import division, print_function

import os
import sys
import csv
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance, XGBClassifier
import eli5
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from eli5.sklearn import PermutationImportance
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

warnings.filterwarnings("ignore")

current_direc = os.getcwd()
print("Current Working Directory is :", current_direc)
## Specify the output directory and the name of the simulation.
outmain = 'task'
outdir = 'Plot_Models'
# label = 'Plot_Models'
outdir = os.path.join(current_direc, outdir)

if os.path.exists(outdir):
    print("{} directory already exist".format(outdir))
else :
    print("{} directory does not exist".format(outdir))
    try:
        os.mkdir(outdir)
    except OSError:
        print("Creation of the directory {} failed".format(outdir))
    else:
        print("Successfully created the directory {}".format(outdir))

## """ Read the data from the provided task_file """
## https://pandas.pydata.org/
task_file = ('./task_data.csv')
with open(task_file,mode='rb') as f:
    dataset = pd.read_csv(f)

## Check the data properties i.e. Identifying the important features within the provided data
## Benifits :
## 1.) help to concentrate on important features thus helping in building the model
## 2.) Get rid of features which are irrelavent and will not contribute to the model performance.

# print(dataset.info())
# print(dataset.describe())

## Split data into training features and labels
## Extracting Labels from the data
Y = dataset['class_label']
## Extracting Features from the data
dataset.drop(['sample index','class_label'],axis=1,inplace=True)
X = dataset

## Further Creating a Correlation Matrix to check out the correlation between the sensors in the dataset
corr_matrix = dataset.corr()

## Heatmap of the correlation matrix
## https://seaborn.pydata.org/
sns.heatmap(corr_matrix, cmap='Blues', annot=True)
plt.savefig(outdir +'/Correlation_Matrix of Sensors.png',dpi=300)
# plt.show()
plt.close()

## split data into train and test dataset
## https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

## Defining the model to rank the sensors according to their importance/predictive power
## with respect to the class labels of the samples
## Testing the Accuracy and Efficiency of model

## https://www.kaggle.com/dansbecker/xgboost
## https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
## https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn


def model(model, model_name, X_train, y_train, X_test, y_test):
    """
    :param model_name:
    :return:
    """
    ## instantiate model
    model = model() #n_estimators=300, learning_rate=0.1)

    ## Training model using training data set
    model.fit(X_train, y_train) #, eval_metric=['error','logloss'], early_stopping_rounds=10,
              #eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

    # print('Model Feature Importance', model.feature_importances_)
    plt.bar(range(len(model.feature_importances_)),model.feature_importances_)
    plt.xlabel('Sensors')
    plt.ylabel('Importance')
    plt.savefig(outdir+'/'+ str(model_name)+'_Feature_Plot.png',dpi=300)
    plt.show()
    plt.close()

    if model_name == XGBClassifier:
        ## Visualizing the importance of each feature/sensor in the data set with their F-values.
        plot_importance(model,importance_type='gain')
        plt.savefig(outdir+'/'+str(model_name)+'_FeatureImportance Plot.png',dpi=300)
        plt.show()
        plt.close()

    ## Ranking of sensors based on their importance value in a reverse order
    rank_sensors = sorted(zip(model.feature_importances_, X.columns), reverse=True)
    for label in rank_sensors:
        print(str(model_name) + ' For {} Predicitve Score is {}'.format(label[1], label[0]))

    ## Seperating features and their importance values
    scores = list(zip(*rank_sensors))[0]
    sensors = list(zip(*rank_sensors))[1]
    sensors_list = np.arange(len(sensors))

    ## Plot features with respect to their importance value
    sns.set()
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(8, 4))
    sns.barplot(sensors_list, np.array(scores), palette="flare")
    plt.xticks(sensors_list, sensors)
    plt.ylabel('Importance Score')
    plt.title(str(model_name))
    # plt.tight_layout()
    plt.savefig(outdir+'/'+str(model_name)+'_Sensors_vs_Importance_plot.png',dpi=300)
    plt.show()
    plt.close()

    ## Save ranked list of sensors and their importance values in a csv file
    ranked_sensors = outdir+'/'+str(model_name)+'_ranked_sensors.csv'
    header = ['Sensor', 'Importance value']

    with open(ranked_sensors,'w') as output:
        ## create the csv writer
        write = csv.writer(output)
        ## write the header
        write.writerow(header)
        # write the row to csv file
        write.writerows(zip(sensors, scores))
    output.close()

    ## Evaluate the predicitions
    predictions = model.predict(X_test)

    ## Mean Absolute error
    print(str(model_name) + " Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))

    ## Model Accuracy i.e how correct is the classifier?
    accuracy = accuracy_score(y_test, predictions)
    print(str(model_name) + ' Accuracy of Model is : %.2f%%'%(accuracy*100.0))

    ## https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
    ## Scalabolity with respect to numbers of features
    ## Fitting model using each importance as threshold

    thresholds = np.sort(model.feature_importances_)
    for threshold in thresholds:
        ## Selecting feature based on threshold value
        select_feature = SelectFromModel(model, threshold = threshold, prefit = True)
        select_X_train = select_feature.transform(X_train)
        ## Train the model for each feature
        select_feature_model = xgb.XGBClassifier(eval_metric='logloss')
        select_feature_model.fit(select_X_train, y_train)
        ## Evaluating the model
        select_X_test = select_feature.transform(X_test)
        ## Evaluating the predicition
        feature_predicition = select_feature_model.predict(select_X_test)
        feature_predicition = [round(value) for value in feature_predicition]
        ## Evaluating the Accuracy
        feature_accuracy = accuracy_score(y_test, feature_predicition)
        print(str(model_name) + ' Threshold = %.5f, n=%d, Accuracy : %.2f%%' % (threshold,select_X_train.shape[1],feature_accuracy*100.0))


## Model 1 : XGBoost
# model_def = model(XGBClassifier, 'XGBoost Classifier', X_train, y_train, X_test, y_test)
## Model 2 : Random Forest
# model_def = model(RandomForestClassifier, 'RandomForest Classifier',  X_train, y_train, X_test, y_test)
## Model 3 : AdaBoost
model_def = model(AdaBoostClassifier, 'AdaBoost Classifier', X_train, y_train, X_test, y_test)
