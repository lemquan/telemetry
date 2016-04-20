#Telemetry
First attempt on predicting the health status of the Sunstone network in our environment. The positive data set consists of a normal stream of data (~600 mbp) and the negative data set is generated from a traffic shaper to reduce flow to ~300. 

We notice that our results had 100% accuracy with Random Forest, 55% with Logistic regression, and 65% with SVM. This means that there is potential information leakage or something is off with our data set.

Using the random forest tree, we establish there are many features unnecessary to the data set as seen in the feature_importance_1,2,3.pngs. Recognising this, we perform dimensionality reduction using SVD. The norm_reduce.png plots the original and the decomposed data set. As you can see, the original data set shows no patterns where as the decomposed shows a sine/cosine like pattern. That is to say, a linear model will not separate the data set but a polynomial relationship like SVM. 

We would prefer to use Scipy's SVD; however, our sparse data set requires more computational resources. We use sklearn's TruncatedSVD. We specifiy the number of components to 17 as that is the rank of our data set. The following shows the performance of Random Forest, Logistic Regression, and SVM. RF performs as we expect with a 95% accuracy whilst logisitc regression is still guessing. SVM performs much better and provides a more realistic prediction. 

## Running the code
The bulk of this code is in test.py. There are several notebooks that contains data analysis, and how we merged the data set from the ELK stack.




## Performance 
We assess the performance of the three classifiers. We train our model using 5 folds cross validation to prevent overfitting. The accuracy uses the F1 score.

### Random Forest with SVD 17 columns 

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=800, n_jobs=1,
            oob_score=False, random_state=45, verbose=0, warm_start=False)
Accuracy for Fold 0
             precision    recall  f1-score   support

          0       0.96      0.95      0.95      2898
          1       0.95      0.96      0.95      2892

avg / total       0.95      0.95      0.95      5790

Accuracy for Fold 1
             precision    recall  f1-score   support

          0       0.96      0.95      0.95      2898
          1       0.95      0.96      0.95      2892

avg / total       0.95      0.95      0.95      5790

Accuracy for Fold 2
             precision    recall  f1-score   support

          0       0.95      0.93      0.94      2897
          1       0.93      0.95      0.94      2891

avg / total       0.94      0.94      0.94      5788

Accuracy for Fold 3
             precision    recall  f1-score   support

          0       0.96      0.94      0.95      2897
          1       0.94      0.96      0.95      2891

avg / total       0.95      0.95      0.95      5788

Accuracy for Fold 4
             precision    recall  f1-score   support

          0       0.96      0.95      0.95      2897
          1       0.95      0.96      0.95      2891

avg / total       0.95      0.95      0.95      5788

### Logistic Regression 

LogisticRegression(C=3.8624137931034483, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=-1, penalty='l2', random_state=45,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
Accuracy for Fold 0
             precision    recall  f1-score   support

          0       0.58      0.60      0.59      2896
          1       0.58      0.56      0.57      2893

avg / total       0.58      0.58      0.58      5789

Accuracy for Fold 1
             precision    recall  f1-score   support

          0       0.57      0.56      0.57      2896
          1       0.57      0.58      0.57      2893

avg / total       0.57      0.57      0.57      5789

Accuracy for Fold 2
             precision    recall  f1-score   support

          0       0.57      0.57      0.57      2896
          1       0.57      0.58      0.57      2893

avg / total       0.57      0.57      0.57      5789

Accuracy for Fold 3
             precision    recall  f1-score   support

          0       0.57      0.57      0.57      2896
          1       0.57      0.57      0.57      2893

avg / total       0.57      0.57      0.57      5789

Accuracy for Fold 4
             precision    recall  f1-score   support

          0       0.58      0.59      0.59      2896
          1       0.58      0.58      0.58      2892

avg / total       0.58      0.58      0.58      5788

accuracy for test:
             precision    recall  f1-score   support

          0       0.58      0.58      0.58      7120
          1       0.58      0.57      0.58      7136

avg / total       0.58      0.58      0.58     14256

### SVM

Accuracy for Fold 0
             precision    recall  f1-score   support

          0       1.00      0.46      0.63      2891
          1       0.65      1.00      0.79      2899

avg / total       0.83      0.73      0.71      5790

Accuracy for Fold 1
             precision    recall  f1-score   support

          0       1.00      0.49      0.66      2891
          1       0.66      1.00      0.80      2898

avg / total       0.83      0.75      0.73      5789

Accuracy for Fold 2
             precision    recall  f1-score   support

          0       1.00      0.48      0.65      2891
          1       0.66      1.00      0.79      2898

avg / total       0.83      0.74      0.72      5789

Accuracy for Fold 3
             precision    recall  f1-score   support

          0       1.00      0.46      0.63      2890
          1       0.65      1.00      0.79      2898

avg / total       0.82      0.73      0.71      5788

Accuracy for Fold 4
             precision    recall  f1-score   support

          0       1.00      0.47      0.64      2890
          1       0.65      1.00      0.79      2898

avg / total       0.83      0.73      0.71      5788

accuracy for test:
             precision    recall  f1-score   support

          0       1.00      0.47      0.64      7147
          1       0.65      1.00      0.79      7109

avg / total       0.83      0.74      0.72     14256

 SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=1)), {}) 948.794399 sec
