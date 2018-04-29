#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 07:46:49 2018

@author: jeremylew
"""


import pandas as pd
import numpy as np
import sys, os
sys.path.append("/Users/jeremylew/Documents/mimic3-benchmarks/")
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

# =============================================================================
# Read data
# =============================================================================


# Modifies the reader for our purpose
class modifiedInHospitalMortalityReader(InHospitalMortalityReader):
    def __init__(self, dataset_dir, listfile=None, period_length=48.0):
        InHospitalMortalityReader.__init__(self, dataset_dir, listfile=None, period_length=48.0)
    
    def _read_timeseries(self, ts_filename):
        ts_df = pd.read_csv(os.path.join(self._dataset_dir, ts_filename))
        assert(ts_df.columns[0]=="Hours")
        subject_id = ts_filename[:ts_filename.find("_timeseries")]
        ts_df["subject_id"] = subject_id
        
        return (ts_df, ts_df.columns)


train_reader = modifiedInHospitalMortalityReader(dataset_dir="../data/in-hospital-mortality/train",
                                             listfile="../data/in-hospital-mortality/train/listfile.csv",
                                             period_length=48.0)

test_reader = modifiedInHospitalMortalityReader(dataset_dir="../data/in-hospital-mortality/test",
                                             listfile="../data/in-hospital-mortality/test/listfile.csv",
                                             period_length=48.0)

# Dictionary of X, y, t..
data_dict_train = common_utils.read_chunk(train_reader, train_reader.get_number_of_examples())
data_dict_test = common_utils.read_chunk(test_reader, test_reader.get_number_of_examples())

class dataTransformer(TransformerMixin):
    def __init__(self):
        self.glascow_coma_scale_eye_opening_mapping = {"Glascow coma scale eye opening": 
                                                        {"1 No Response": "No Response", 
                                                         "2 To pain": "To Pain", 
                                                         "3 To speech": "To Speech", 
                                                         "4 Spontaneously": "Spontaneously"},
                                                      "Glascow coma scale motor response":
                                                        {"1 No Response": "No response",
                                                         "2 Abnorm extensn": "Abnormal extension",
                                                         "3 Abnorm flexion": "Abnormal Flexion",
                                                         "4 Flex-withdraws": "Flex-withdraws",
                                                         "5 Localizes Pain": "Localizes Pain",
                                                         "6 Obeys Commands": "Obeys Commands"},
                                                       "Glascow coma scale verbal response":
                                                        {"No Response-ETT": "No Response",
                                                         "1 No Response": "No Response",
                                                         "1.0 ET/Trach": "No Response",
                                                         "2 Incomp sounds": "Incomprehensible sounds",
                                                         "3 Inapprop words": "Inappropriate Words",
                                                         "4 Confused": "Confused",
                                                         "5 Oriented": "Oriented"}}
                           
    def transform(self, data_dict, y=None):
        raw_df = pd.concat(data_dict["X"])
        print(raw_df.dtypes)
        
        # Reset indexes after concatenating dataframes
        raw_df = raw_df.reset_index(drop=True)
        raw_df["id"] = raw_df.index
        
        raw_df = raw_df.replace(to_replace=self.glascow_coma_scale_eye_opening_mapping)
        raw_df["Glascow coma scale eye opening"] = raw_df["Glascow coma scale eye opening"].fillna("ZZNAN").astype(pd.api.types.CategoricalDtype(categories=["None", "No Response", "To Pain", "To Speech", "Spontaneously", "ZZNAN"]))
        raw_df["Glascow coma scale motor response"] = raw_df["Glascow coma scale motor response"].fillna("ZZNAN").astype(pd.api.types.CategoricalDtype(categories=["No response","Abnormal extension","Abnormal Flexion","Flex-withdraws","Localizes Pain", "Obeys Commands","ZZNAN"]))
        raw_df["Glascow coma scale verbal response"] = raw_df["Glascow coma scale verbal response"].fillna("ZZNAN").astype(pd.api.types.CategoricalDtype(categories=["No Response","Incomprehensible sounds", "Inappropriate Words", "Confused", "Oriented","ZZNAN"]))
        
        
        glascow_eye_opening_le = preprocessing.LabelEncoder()
        glascow_eye_opening_le.fit(raw_df["Glascow coma scale eye opening"])
        raw_df["Glascow coma scale eye opening"] = glascow_eye_opening_le.transform(raw_df["Glascow coma scale eye opening"])
        
        glascow_motor_response_le = preprocessing.LabelEncoder()
        glascow_motor_response_le.fit(raw_df["Glascow coma scale motor response"])
        raw_df["Glascow coma scale motor response"] = glascow_motor_response_le.transform(raw_df["Glascow coma scale motor response"])
        
        glascow_verbal_response_le = preprocessing.LabelEncoder()
        glascow_verbal_response_le.fit(raw_df["Glascow coma scale verbal response"])
        raw_df["Glascow coma scale verbal response"] = glascow_verbal_response_le.transform(raw_df["Glascow coma scale verbal response"])
        
        # Convert data to long format
        cols = [c for c in raw_df.columns if c not in ["subject_id", "Hours", "id"]]
        raw_df = pd.melt(raw_df,id_vars=["id","subject_id","Hours"], value_vars=cols)
        
        # Exclude NaN values
        raw_df = raw_df[~((raw_df["variable"]=="Glascow coma scale eye opening") & (raw_df["value"]==glascow_eye_opening_le.transform(["ZZNAN"])[0]))]
        raw_df = raw_df[~((raw_df["variable"]=="Glascow coma scale motor response") & (raw_df["value"]==glascow_motor_response_le.transform(["ZZNAN"])[0]))]
        raw_df = raw_df[~((raw_df["variable"]=="Glascow coma scale verbal response") & (raw_df["value"]==glascow_verbal_response_le.transform(["ZZNAN"])[0]))]
        raw_df = raw_df[raw_df["value"].notnull()].drop(columns=["id"]).reset_index(drop=True)
        raw_df.value.apply(type).value_counts()

        return raw_df


d = dataTransformer()
df = d.transform(data_dict_train)
df_test = d.transform(data_dict_test)

# =============================================================================
# Feature extraction
# =============================================================================

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
settings = MinimalFCParameters()
#settings = EfficientFCParameters()


def extractTSFeatures(df, data_dict, settings):
    features = extract_features(df,column_id="subject_id",column_sort="Hours",column_kind="variable", column_value="value",default_fc_parameters=settings)
        
    return features


fc_parameters = {
    "abs_energy": None,
    "absolute_sum_of_changes": None,
    "agg_autocorrelation": [{"f_agg": "mean"},{"f_agg": "var"}],
#    "ar_coefficient": [{"coeff": 2, "k": 2}]
    "autocorrelation": [{"lag":1}],
    "count_above_mean": None,
    "count_below_mean": None,
    "kurtosis": None,
    "skewness": None,
    "sample_entropy": None,
    "maximum": None,
    "mean": None,
    "mean_abs_change": None,
    "mean_change": None,
    "median": None,
    "minimum": None,
    "standard_deviation": None,
    "variance": None,
    "sum_of_reoccurring_values": None,
    "sum_values": None
}

#features_train = extractTSFeatures(df, data_dict_train, fc_parameters)
#features_test = extractTSFeatures(df_test, data_dict_test, fc_parameters)

features_train = pd.read_csv("Features dataset/features_train.csv")
def orderLabels(features,data_dict):
    labels = pd.DataFrame(list(zip(data_dict["name"],data_dict["y"])),columns=["name","y"])
    labels.name = labels.name.apply(lambda x: x[:x.find("_timeseries.csv")])
    new = features_train.merge(labels,how="left", left_on="id", right_on="name") 
    assert(new.shape[0]==features.shape[0])
    
    return new.y.values #labels


y_train = orderLabels(features_train, data_dict_train)
features_train = features_train.drop(columns=["id"])


# Check which columns have too many missing values
column_density = np.array([features_train[c].count()/features_train.shape[0] for c in features_train.columns])
filtered_columns = features_train.columns[np.where(column_density > 0.5)]
X_train = features_train[filtered_columns]
            
# Impute missing values
from sklearn.preprocessing import Imputer
imputer = Imputer()
X_train = imputer.fit_transform(X_train.values)

# Standardize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Select best features
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
rr_model = Ridge(alpha=2)
rfe = RFE(estimator=rr_model, n_features_to_select=50)
rfe.fit(X_train,y_train)
X_train = X_train[:,rfe.support_]



# =============================================================================
# Fit logistic regression model
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE


penalty = ["l2"]
regularisation = [0.1, 1, 10, 100, 1000]

def chooseOptimalParameters(penalty, regularisation, X_train, y_train):
    auroc = {}
    auprc = {}
    
    for p in penalty:
        for r in regularisation:
            auroc[(p,r)] = []
            auprc[(p,r)] = []
            logisticModel = LogisticRegression(penalty=p, C=r, solver="newton-cg",max_iter=1000)
            print(logisticModel)
            
            kf = StratifiedKFold(n_splits=5)
            for train_fold_idx, cv_fold_idx in kf.split(X_train, y_train):
                # Smote 
                sm = SMOTE(random_state=40)
                X_train_res, y_train_res = sm.fit_sample(X_train[train_fold_idx], y_train[train_fold_idx])
                
                logisticModel.fit(X_train_res, y_train_res) 
                y_pred = logisticModel.predict(X_train[cv_fold_idx])
                
                # AUPRC
                precision, recall, _ = precision_recall_curve(y_train[cv_fold_idx], y_pred)
                auprc[(p,r)].append(auc(recall,precision))
                
                # AUROC 
                fpr, tpr, _ = roc_curve(y_train[cv_fold_idx], y_pred)
                auroc[(p,r)].append(auc(fpr,tpr))
   
    return auroc, auprc


auroc, auprc = chooseOptimalParameters(penalty, regularisation, X_train, y_train)




