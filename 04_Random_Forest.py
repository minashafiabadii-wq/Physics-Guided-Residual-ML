from sklearn.ensemble import RandomForestRegressor

def build_model(random_state=42):
    return RandomForestRegressor(n_estimators=30, max_depth=18, min_samples_leaf=2, max_features='sqrt', random_state=random_state, n_jobs=-1)
