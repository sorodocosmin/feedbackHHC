from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def grid_search(parameters, model, x_train, y_train):
    """
    It will perform a grid search to find the best hyperparameters for the model
    :return: a dictionary with the best hyperparameters
    """
    grid_src = GridSearchCV(estimator=model, param_grid=parameters, cv=10, n_jobs=-1, verbose=1)
    grid_src.fit(x_train, y_train)

    return grid_src.best_params_, grid_src.best_score_


def random_search(parameters, model, x_train, y_train):
    """
    It will perform a random search to find the best hyperparameters for the model
    :return: a dictionary with the best hyperparameters
    """
    grid_src = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=10, n_jobs=-1, verbose=1)
    grid_src.fit(x_train, y_train)

    return grid_src.best_params_, grid_src.best_score_
