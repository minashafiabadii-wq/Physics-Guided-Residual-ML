from catboost import CatBoostRegressor

def build_model(random_state=42):
    return CatBoostRegressor(iterations=30, depth=6, learning_rate=0.07, loss_function='RMSE', random_seed=random_state, verbose=False, allow_writing_files=False, thread_count=4)
