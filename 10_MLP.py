from sklearn.neural_network import MLPRegressor

def build_model(random_state=42):
    return MLPRegressor(hidden_layer_sizes=(48,24), activation='relu', solver='adam', alpha=0.001, learning_rate_init=0.001, max_iter=80, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=random_state)
