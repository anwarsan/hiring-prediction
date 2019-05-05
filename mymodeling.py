import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from sklearn.utils.fixes import signature

"""
def myscore_thr(y,ypred_proba):
    thresh = 0.3
    ypred_thr = (ypred_proba > thresh).astype(int)
    return f1_score(y, ypred_thr)"""
       
def clf_gs(clf, Xtrain, ytrain, Xtest, ytest):
    """
    
         Grid Search with Cross Validation on nbsplit K-folds
        
            * params: 
               - clf: classifier
               - Xtrain, ytrain, Xtest, ytest
          
            * returns:
               - clf_gs: best classifier
    
    """    
    nbsplit = 3
    cv = StratifiedShuffleSplit(n_splits=nbsplit, test_size=0.3, random_state=20)
    params = {'n_estimators': [500, 1000],
              'max_depth': [20, 80],#, None], 
              'min_samples_leaf': [5, 10],
              'min_samples_split': [2, 5],
              'class_weight': ['balanced_subsample'],# 'balanced',
              'random_state': [112]
             }  

    # average_precision: Area under PR curve (for binary classification)
    score = {'F1':'f1','AUC-PR':'average_precision'} 
    print("Training... (scoring:",score,")\n")    
    clf_gs = GridSearchCV(clf, 
                        params, 
                        scoring = score,refit='AUC-PR',
                        cv = cv,
                        n_jobs=-1)
    clf_gs.fit(Xtrain, ytrain)

    best_params = clf_gs.best_estimator_
    print("Best Parameters: ", best_params)
    print("Best Score: %.2f%%" % ((clf_gs.best_score_)*100))
    
    return clf_gs     


def clf_print_results(clf, Xtrain, ytrain, Xtest, ytest):
    """
    
        Classifier results:   
        
          * params: 
              - clf: classifier
              - Xtrain, ytrain, Xtest, ytest
          * prints:
              - classification report 
              - accuracy score
              - confustion matrix
              - AUC for ROC and ROC curve
              - AUC for precision recall and precision-recall curve    
              
    """
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    print("Classification report:\n", classification_report(ytest,ypred))
    accuracy = accuracy_score(ytest, ypred) 
    conf_mat = confusion_matrix(ytest, ypred)  
    print("Accuracy %.2f%%" % (accuracy*100)) 
    print("---- Confusion matrix ----")
    print(conf_mat)
    
    ypred_proba = clf.predict_proba(Xtest)[:,1]
    fpr, tpr, thr_roc = roc_curve(ytest, ypred_proba)
    auc_roc = auc(fpr,tpr)
    precision, recall, thr_pr = precision_recall_curve(ytest, ypred_proba)
    auc_pr = auc(recall, precision)
    average_precision = average_precision_score(ytest, ypred_proba)
    
    #----------------- FIGURES AUC -----------------
    fig, ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].plot(fpr, tpr, '-', lw=2, label='AUC-ROC=%.3f' % auc_roc)
    ax[0].plot([0, 1], [0, 1],linestyle='--')
    ax[0].set_xlabel('False Positive Rate', fontsize=16)
    ax[0].set_ylabel('True Positive Rate', fontsize=16)
    ax[0].set_title('ROC Curve',fontsize=16)
    ax[0].legend(loc="lower right", fontsize=14)
    plt.subplots_adjust(hspace = 0.3, top=0.92) 
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post', label='AUC-PR=%.3f' % auc_pr)
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    ax[1].set_xlabel('Recall', fontsize=16)
    ax[1].set_ylabel('Precision', fontsize=16)
    ax[1].set_title('Precision-Recall Curve', fontsize=16) 
    #ax[1].plot([0, 1], [0.1, 0.1], linestyle='--')
    ax[1].legend(loc="upper right", fontsize=14)
    plt.show()
    

def clf_roc_pr(clf, Xtrain, ytrain, Xtest, ytest):
    """
         Classifier Aera Under Curves:   
        
          * params: 
              - clf: classifier
              - Xtrain, ytrain, Xtest, ytest
          * returns:
              - AUC for ROC 
              - AUC for precision recall 
    
    """
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)  
    ypred_proba = clf.predict_proba(Xtest)[:,1]
    fpr_proba, tpr_proba, _ = roc_curve(ytest, ypred_proba)
    precision, recall, thresholds = precision_recall_curve(ytest, ypred_proba)
    
    return auc(fpr_proba,tpr_proba), auc(recall, precision)