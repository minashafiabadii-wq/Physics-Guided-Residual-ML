from sklearn.neighbors import KNeighborsRegressor

def build_model(random_state=42):
    return KNeighborsRegressor(n_neighbors=15, weights='distance', p=2, n_jobs=-1)
