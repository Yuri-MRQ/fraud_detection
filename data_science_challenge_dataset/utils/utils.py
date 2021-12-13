import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve, auc, 
                             confusion_matrix, precision_recall_curve)



def split_data(X, Y, test_size=0.2, kfold=None, random_state=0):
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,  Y, test_size = test_size,  
                                                        stratify=Y, random_state=random_state)
    
    if kfold:
        
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)

        kfold_5 = skf.split(X_train, Y_train)
        
        i=0
        indices_split = {}

        for train, val in kfold_5:

            indices_split[f"fold_{i}"] = (train, val)

            i += 1
        
        return indices_split, (X_test, Y_test)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_train,  Y_train, test_size = test_size,
                                                      stratify=Y_train, random_state=RANDOM_STATE)
    
    return (X_train, X_val, X_test), (Y_train, Y_val, Y_test)

def compare_models(X, Y, classifiers=None, indices_split=None, scaling=None):
    """
    Args:
    
    X, Y - if calssifiers == None and indices_split == None, X and Y are tuples
    that contains (X_train, X_val) e (Y_train, Y_test), otherwise X, Y
    are np.array of all dataset to be separeted using the indices_split
    into n fold.
    classifiers - dictionary of classifiers or a single classifier.
    
    
    """
    history = {}
    
    if indices_split and classifiers:

        for clf_name, clf, (train, val) in zip(classifiers.keys(), classifiers.values(), indices_split.values()):

            print(f"\n Evaluating the model {clf_name}")
            
            if scaling:
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X[train])
                X_val = scaler.transform(X[val])
            else:
                X_train = X[train]
                X_val = X[val]

            #Training 

            y_hat_train = clf.predict(X_train)

            rf_train_roc_auc_score = roc_auc_score(Y[train], y_hat_train)
            rf_train_f1_score = f1_score(Y[train], y_hat_train, average="weighted")

            #val

            y_hat_val = clf.predict(X_val)

            rf_val_roc_auc_score = roc_auc_score(Y[val], y_hat_val)
            rf_val_f1_score = f1_score(Y[val], y_hat_val, average="weighted")
            precision, recall, thresholds_pr = precision_recall_curve(Y[val], y_hat_val)
            AUC_PRcurve= auc(recall, precision)
            fpr, tpr, thresholds = roc_curve(Y[val], y_hat_val)

            print(f"F1 score - Train: {rf_train_f1_score} / val: {rf_val_f1_score}")
            print(f"ROC AUC score - Train: {rf_train_roc_auc_score} / val: {rf_val_roc_auc_score}")
            print(f"PR AUC score - val: {AUC_PRcurve}")

            history[clf_name] = {"y_hat_train": y_hat_train, "y_hat_val": y_hat_val,
                                 "pr": (precision, recall, AUC_PRcurve), "roc": (fpr, tpr, thresholds)}
    else:
        clf = classifiers
        #Train
        X_train, X_val = X
        y_train, y_val = Y
        
        if scaling:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.fit(X_val)
        
        if X_train is not None:

            y_hat_train = clf.predict(X_train)

            rf_train_roc_auc_score = roc_auc_score(y_train, y_hat_train)
            rf_train_f1_score = f1_score(y_train, y_hat_train, average="weighted")

        #val

        y_hat_val = clf.predict(X_val)

        rf_val_roc_auc_score = roc_auc_score(y_val, y_hat_val)
        rf_val_f1_score = f1_score(y_val, y_hat_val, average="weighted")
        precision, recall, thresholds_pr = precision_recall_curve(y_val, y_hat_val)
        AUC_PRcurve= auc(recall, precision)
        fpr, tpr, thresholds = roc_curve(y_val, y_hat_val)
        
        if X_train is not None:
            print(f"F1 score - Train: {rf_train_f1_score} / val: {rf_val_f1_score}")
            print(f"ROC AUC score - Train: {rf_train_roc_auc_score} / val: {rf_val_roc_auc_score}")
            print(f"PR AUC score - val: {AUC_PRcurve}")
        else:
            y_hat_val = clf.predict(X_val)

            rf_val_roc_auc_score = roc_auc_score(y_val, y_hat_val)
            rf_val_f1_score = f1_score(y_val, y_hat_val, average="weighted")
            precision, recall, thresholds_pr = precision_recall_curve(y_val, y_hat_val)
            AUC_PRcurve= auc(recall, precision)
            fpr, tpr, thresholds = roc_curve(y_val, y_hat_val)
            
            print(f"ROC AUC score val: {rf_val_roc_auc_score}")
            print(f"PR AUC score val: {AUC_PRcurve}")
            
            history["clf_0"] = {"y_hat_test": y_hat_val,
                                       "pr": (precision, recall, AUC_PRcurve), 
                                       "roc": (fpr, tpr, thresholds)}
            
            return history
            

        history["clf_0"] = {"y_hat_train": y_hat_train, "y_hat_val": y_hat_val,
                            "pr": (precision, recall, AUC_PRcurve), "roc": (fpr, tpr, thresholds)}
        
    return history


def plot_pr_curve(history):

    plt.figure(1)

    for model_name, model_eval in zip(history.keys(), history.values()):

        precision, recall, AUC_PRcurve = model_eval["pr"]

        # plot no skill
        plt.plot([0, 1], [0.5, 0.5], linestyle='--')
        #plot PR curve
        plt.plot(precision, recall, label = "{} AUC = {:0.2f}".format(model_name, AUC_PRcurve), lw = 3, alpha = 0.7)
    plt.xlabel('Precision', fontsize = 14)
    plt.ylabel('Recall', fontsize = 14)
    plt.title('Precision-Recall Curve', fontsize = 18)
    plt.legend(loc='best')
    plt.show()

def plot_roc_auc_curve(history):
    
    plt.figure()
    
    for model_name, model_eval in zip(history.keys(), history.values()):

        fpr, tpr, thresholds = model_eval["roc"]
        
        roc_auc = auc(fpr, tpr)

        lw = 2
        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    
    plt.show()

def split_indices_stritify(y, proportion, rate_positive):
 
    """
    Args:
    
    proportion - is the proportion of the original data we want to slice
    rate_positive - rate of the positive label of the imbaleced data
    
    """
    array_size = len(y)
    
    new_size = proportion*array_size
    
    positive_size = int(rate_positive * new_size)
    negative_size = int((1 - rate_positive) * new_size)
    
    positive_indicies = np.argwhere(Y==1)[0:positive_size]
    negative_indices = np.argwhere(Y==0)[0:negative_size]
    
    return positive_indicies, negative_indices

def plot_cm(labels, predictions, model_name):
    print(model_name)
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def train_model(model, X, Y, indices_split=None, scaling=False):
    
        
    if indices_split:
        i = 0
        classifiers = {}
        for train, val in indices_split.values():
            
            if scaling:
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X[train])
            else:
                X_train = X[train]
                
            print(f"Training clf {i}")

            model.fit(X_train, Y[train])

            classifiers[f"clf_{i}"] = model

            i += 1
        return classifiers
    
    if scaling:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
    model.fit(X, Y)
    
    return model
    
def pr_auc_score(y_true, y_score):
    """
    Generates the Area Under the Curve for precision and recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

