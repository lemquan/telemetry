import pandas as pd
import numpy as np
import sys
import time
import pickle 
from sklearn import cross_validation, linear_model, metrics, ensemble, grid_search, svm, decomposition
from scipy import interp
from pprint import pprint

if sys.platform == 'darwin':
    import matplotlib as mil
    mil.use('TkAgg')

    import matplotlib.pyplot as plt
    plot_on = True
    print "Running OS X"

elif sys.platform == 'linux' or sys.platform == 'linux2':
    print "Running Linux. Plots are saved."
    import matplotlib as mil
    mil.use('Agg')
    import matplotlib.pyplot as plt
    plot_on = False

def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        res = func(*args, **kwargs)
        te = time.time()

        print '%r (%r, %r) %f sec' % \
        			(func.__name__, args, kwargs, te-ts)
        return res
    return timed

def load(d):
    df = pd.read_csv('data/final.csv', index_col=0)
    df = df.sample(frac=1)

    y = df[['health']].as_matrix().reshape(-1)
    df = df.drop('health', 1)
    features = list(df)

    if d == 1:
        df = df.drop(['ThrottledPacketsReceived', 'CRCErrors', 'SecondsSinceLastClearCounters', \
                      'OutputQueueDrops', 'OutputUnderruns', 'InputErrors', 'AvailabilityFlag', \
                      'InputDrops', 'OutputDrops', 'OutputBuffersSwappedOut', 'Resets', 'InputAborts', \
                      'GiantPacketsReceived', 'FramingErrorsReceived', 'OutputBufferFailures', \
                      'CarrierTransitions', 'RuntPacketsReceived', 'InputQueueDrops', 'InputOverruns', \
                      'OutputErrors', 'Applique', 'InputIgnoredPackets', 'MulticastPacketsSent', \
                      'MulticastPacketsReceived', 'ParityPacketsReceived', 'UnknownProtocolPacketsReceived', \
                      'PacketsReceived', 'PacketsSent', 'BytesReceived', 'BytesSent', \
                      'delta.PacketsReceived', 'delta.PacketsSent', 'delta.BytesSent', \
                      'delta.BytesReceived', 'epoch_time', 'LastDataTime', 'LastDiscontinuityTime',\
                      'ip', 'hostname'], 1)
        features = list(df)

    # pprint (list(df))
    X = df.as_matrix()

    svd = decomposition.TruncatedSVD(n_components=17, random_state=55)
    trans_X = svd.fit_transform(X)

    # plotting the decomposed and original data
    # pos_idx = np.where(y == 1)
    # neg_idx = np.where(y == 0)
    #
    # X_pos = trans_X[pos_idx]
    # X_neg = trans_X[neg_idx]
    #
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25,25))
    # ax1.plot(X[pos_idx][:,18], X[pos_idx][:,14], 'r+')
    # ax1.plot(X[neg_idx][:,18], X[neg_idx][:,14], 'go')
    # ax1.set_title('Original Data')
    # ax2.plot( X_pos[:,0], X_pos[:,11], 'r+', label='positive')
    # ax2.plot(X_neg[:,0], X_neg[:,11], 'go', label='negative')
    # ax2.set_title('Dimensionality Reduced')
    # plt.show()
    #f.savefig('norm_reduced.png')

    return trans_X, y, features


def split(X, y):
    # split the data set into 80/20
    X_train, X_test, y_train, y_test = cross_validation.train_test_split \
        (X, y, test_size=0.33, random_state=42)
    p = np.where(y_test == 1)
    n = np.where(y_test == 0)

    p2 = np.where(y_train == 1)
    n2 = np.where(y_train == 0)

    print 'train', 'pos:', len(p2[0]), 'neg:', len(n2[0]), 'size', X_train.shape
    print 'test', 'pos:', len(p[0]), 'neg:', len(n[0]), 'size', X_test.shape

    return X_train, X_test, y_train, y_test


def find_feat_importance(X_train, y_train, X_test, y_test, features):
    print 'finding important features using random forest....'
    clf = ensemble.RandomForestClassifier(n_estimators=700, max_features='log2', criterion='entropy', random_state=45)
    clf = clf.fit(X_train, y_train)
    print metrics.classification_report(y_test, clf.predict(X_test))


    # plot the important features
    f = 100. * (clf.feature_importances_ / clf.feature_importances_.max())
    sorted_idx = np.argsort(f)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.figure(figsize=(16, 12))
    plt.barh(pos, f[sorted_idx], align='center')
    plt.yticks(pos, np.asanyarray(features)[sorted_idx])
    plt.title('Important Features')

    plt.savefig('feature_importances_3.png')
    if plot_on == True:
        plt.show()

def create_model(X_train, y_train, model='log_reg'):
    if model == 'random_forest':
        print 'creating the random forest model....'
        clf = ensemble.RandomForestClassifier(random_state=45)
        params = {'n_estimators': [10, 100, 500, 800], 'criterion':['gini', 'entropy']}

    elif model == 'svm':
        print 'creating svm...'
        clf = svm.SVC(verbose=1)
        params = {'kernel':['rbf'], 'C': [0.01, 1, 1.5]}
    else:
        # default to logistic regression
        print 'creating the log reg model....'
        clf = linear_model.LogisticRegression(random_state=45, n_jobs=-1)
        params = {'C': np.logspace(0.001, 1.5, 40)}

    # parameter search
    print 'running grid search....'
    scoring = None
    if model is not 'svm':
        scoring = 'f1'
    grid = grid_search.GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring=scoring) # score='f1'
    grid.fit(X_train, y_train)

    print 'best estimator parameters:'
    print grid.best_estimator_
    return grid.best_estimator_

@timeit
def train_model(X, y, X_test, y_test, clf, folds=5):
    strat_k_fold = cross_validation.StratifiedKFold(y, n_folds=folds, shuffle=True, random_state=45)

    y_hats = []
    for train_idx, valid_idx in strat_k_fold:
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train = y[train_idx]
        clf = clf.fit(X_train, y_train)
        y_hats.append((y[valid_idx], clf.predict(X_valid)))

    # assess the accuracy of validation
    mean_tpr = 0.0
    mean_fpr = np.linspace(0,1, 100)
    all_tpr = []
    fig = plt.figure(figsize=(10, 8))
    for i, (y_valid, y_hat) in enumerate(y_hats):
        print 'Accuracy for Fold', i
        print metrics.classification_report(y_valid, y_hat)

         # plot the ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_hat)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC Fold %d (area = %0.02f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color='0.75', label='Random Guess')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

    plt.savefig('roc_svm_f1.png')
    if plot_on == True:
        plt.show()

    # predict on test
    y_test_preds = clf.predict(X_test)
    print 'accuracy for test:'
    print metrics.classification_report(y_test, y_test_preds)


if __name__ == '__main__':
    # find the important features
    X, y, features = load(d=1)
    #X_train, X_test, y_train, y_test = split(X,y)
    #find_feat_importance(X_train, y_train, X_test, y_test, features)
    #create_model(X_train, y_train, model='log_reg')

    #clf = create_model(X_train, y_train, model='svm')
    #train_model(X_train, y_train, X_test, y_test, clf)
