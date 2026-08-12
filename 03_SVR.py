from sklearn.svm import SVR

def build_model(random_state=42):
    return SVR(C=5.0, epsilon=0.015, gamma='scale', kernel='rbf', cache_size=1200)
