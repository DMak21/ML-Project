import numpy as np
import copy


class Stack():
    """
    Parameters
    ----------
    models : list 
        List of 1-st level models
    X_train : numpy array
    y_train : numpy 1d array
    X_test : numpy array
    needs_proba : boolean
        Default False
    n_folds : int
        Default 4

    Functions
    -------
    stack
        returns [S_train, S_test]
    """
    def __init__(models, X_train, y_train, X_test,
                 needs_proba=False, n_folds=4):
        self.models = models          
        self.X_train = X_train          
        self.y_train = y_train                  
        self.X_test = X_test      
        self.needs_proba = needs_proba
        self.n_folds = n_folds 


    def model_action(model, X_train, y_train, X_test, action=None):
        if 'fit' == action:
            return model.fit(X_train, y_train)
        elif 'predict' == action:
            return model.predict(X_test)
        elif 'predict_proba' == action:
            return model.predict_proba(X_test)
        else:
            raise ValueError('Parameter action must be set properly')

    def model_params(model):
        s = ''
        
        if hasattr(model, 'get_params'):
            params_dict = model.get_params()
            max_len = 0
            for key in params_dict:
                if len(key) > max_len:
                    max_len = len(key)
            sorted_keys = sorted(params_dict.keys())
            for key in sorted_keys:
                s += '%-*s %s\n' % (max_len, key, params_dict[key])
                
        elif hasattr(model, '__repr__'):
            s = model.__repr__()
            s += '\n'
        
        else:
            s = 'Model has no ability to show parameters (has no <get_params> or <__repr__>)\n'
            
        s += '\n'
            
        return s

    def kf_split(X, n_splits):
        indices = np.arange(X.shape[0])
        fold_sizes = np.full(n_splits, X.shape[0] // n_splits, dtype=np.int)
        fold_sizes[:X.shape[0] % n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size

            test_index = indices[start:stop]
            train_indices = np.concatenate([indices[0:start], indices[stop:indices.shape[0]]])

            yield (train_indices, test_index)
            current = stop

    def stack(self):

        if 0 == len(self.models):
            raise ValueError('List of models is empty')
                                     
        needs_proba = bool(self.needs_proba)

        if not isinstance(self.n_folds, int):
            raise ValueError('Parameter <n_folds> must be integer')
        if not self.n_folds > 1:
            raise ValueError('Parameter <n_folds> must be not less than 2')

        if needs_proba:
            n_classes = len(np.unique(self.y_train))
            action = 'predict_proba'
        else:
            n_classes = 1
            action = 'predict'

        S_train = np.zeros(( self.X_train.shape[0], len(self.models) * n_classes ))
        S_test = np.zeros(( self.X_test.shape[0], len(self.models) * n_classes ))

        models_folds_str = ''
        
        for model_counter, model in enumerate(self.models):

            S_test_temp = np.zeros((self.X_test.shape[0], self.n_folds * n_classes))
            
            # Create empty array to store scores for each fold (to find mean)
            scores = np.array([])

            for fold_counter, (tr_index, te_index) in enumerate(kf_split(self.X_train, self.n_folds)):
                # Split data and target
                X_tr = self.X_train[tr_index]
                y_tr = self.y_train[tr_index]
                X_te = self.X_train[te_index]
                y_te = self.y_train[te_index]

                model = copy.deepcopy(model)

                _ = model_action(model, X_tr, y_tr, None, action = 'fit')

                if 'predict_proba' == action:
                    col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
                else:
                    col_slice_model = model_counter
                S_train[te_index, col_slice_model] = model_action(model, None, None, X_te, action = action)


            _ = model_action(model, self.X_train, self.y_train, None, action = 'fit')
            if 'predict_proba' == action:
                col_slice_model = slice(model_counter * n_classes, model_counter * n_classes + n_classes)
            else:
                col_slice_model = model_counter
            S_test[:, col_slice_model] = model_action(model, None, None, self.X_test, action = action)
                

        if not needs_proba:
            if S_train is not None:
                S_train = S_train.astype(int)
            if S_test is not None:
                S_test = S_test.astype(int)
        return (S_train, S_test)

