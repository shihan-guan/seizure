from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def initialize_svm_model(C=1.0, kernel='linear'):
    return make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel, probability=True))
