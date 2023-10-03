from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC

def get_dt(x, y):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x, y)
    return clf


def get_random_forest(x, y):
    clf = RandomForestClassifier(min_samples_split=0.05)
    clf.fit(x, y)
    return clf


def get_gbc(x, y):
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(x, y)
    return clf


def get_bagging(x, y):
    clf = BaggingClassifier()
    clf.fit(x, y)
    return clf


def get_votting(x, y):
    estimator = [
        ('svm', SVC(probability=True)),
        ('rf', RandomForestClassifier()),
        ('ada', AdaBoostClassifier()),
    ]
    clf = VotingClassifier(estimator)
    clf.fit(x, y)
    return clf


def get_ada(x, y):
    clf = AdaBoostClassifier()
    clf.fit(x, y)
    return clf


def get_his(x, y):
    clf = HistGradientBoostingClassifier()
    clf.fit(x, y)
    return clf


def get_stacking(x, y):
    estimator = [
        ('svm', SVC(probability=True)),
        ('rf', RandomForestClassifier()),
        ('ada', AdaBoostClassifier()),
    ]
    clf = StackingClassifier(estimators=estimator)
    clf.fit(x, y)
    return clf


def get_extree(x, y):
    clf = ExtraTreeClassifier()
    clf.fit(x, y)
    return clf


def get_svc(x, y):
    clf = SVC()
    clf.fit(x, y)
    return clf


METHOD_MAP = {
    "dt": get_dt,
    "random_forest": get_random_forest,
    "gradient_boosting_tree": get_gbc,
    "bagging": get_bagging,
    "votting": get_votting,
    "ada": get_ada,
    "his": get_his,
    "stacking": get_stacking,
    "extree": get_extree,
    "svc": get_svc
}


def get_model(x, y, model="dt"):
    return METHOD_MAP[model](x, y)