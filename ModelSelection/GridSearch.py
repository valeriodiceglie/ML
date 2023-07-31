import itertools

class GridSearch:

    def __init__(self, model, param_grid, cv, metric='MAE'):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.metric = metric
        self.best_score_ = None
        self.best_params_ = None

    def grid_search(self, x_train, y_train, x_test, y_test):
        """
        Perform Grid Search for a given model.

        Parameters:
            x_train (array-like, shape (n_samples, n_features)): The X training samples.
            x_test
            y_train (array-like, shape (n_samples,)): The target vector.
            y_test
        Returns:
            dict: A dictionary containing the best hyperparameters found and their corresponding
                  cross-validated mean score.
        """

        # Generate all combinations of hyperparameter values
        param_combinations = list(itertools.product(*self.param_grid.values()))

        # Perform Grid Search 
        for params in param_combinations:
            param_dict = {param_name: param_value for param_name, param_value in zip(self.param_grid.keys(), params)}
            model_instance = self.model(**param_dict)
            model_instance.fit(x_train, y_train)
            prediction = model_instance.predict(x_test)

            # calculate scores
            mae, mse, rmse, r2 = model_instance.get_scores(prediction, y_test)

            # Check if the current combination of hyperparameters gives a better score
            if self.metric == 'MAE':
                if self.best_score_ is None or mae > self.best_score_:
                    self.best_score_ = mae
                    self.best_params_ = param_dict

            if self.metric == 'MSE':
                if self.best_score_ is None or mse > self.best_score_:
                    self.best_score_ = mse
                    self.best_params_ = param_dict

            if self.metric == 'RMSE':
                if self.best_score_ is None or rmse > self.best_score_:
                    self.best_score_ = rmse
                    self.best_params_ = param_dict

            if self.metric == 'R2':
                if self.best_score_ is None or r2 > self.best_score_:
                    self.best_score_ = r2
                    self.best_params_ = param_dict

        return {'best_params': self.best_params_, 'best_score': self.best_score_}


