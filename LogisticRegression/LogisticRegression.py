import numpy as np

class LogisticRegression(object):
    """
    Parameters
    ----------
    learning_rate : float
        Default: 0.1
    num_steps : int
        Default: 100
    regulariser : string
        None or 'l1’ or ‘l2’ Default:'l2'
    lamda : float
        Default 1
    n_classes : int
        Default: np.max(y) + 1
    initial_wts : numpy array
        Default: np.random.normal(loc=0.0, scale=scale)
        
    Functions
    -------
    train
        returns model
    predict
        returns class labels
    predict_prob
        returns class probabilities
    """
    
    def __init__(self, learning_rate=0.1, num_steps=100,
                 regulariser='l2',
                 lamda=1,
                 n_classes=None,
                 initial_wts=None):

        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.regulariser = regulariser
        self.lamda = lamda
        self.n_classes = n_classes
        self.initial_wts = initial_wts

    def _train(self, X, y, init_params=True):
        if init_params:
            if self.n_classes is None:
                self.n_classes = np.max(y) + 1
            self._n_features = X.shape[1]

            
            self.b_, self.w_ = self._init_params(
                weights_shape=(self._n_features, self.n_classes),
                bias_shape=(self.n_classes,))
            
            if self.initial_wts:
                self.b_, self.w_ = np.split(self.initial_wts, [1])
                
            self.cost_ = []

        y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float)

        for i in range(self.num_steps):
            for idx in self._yield_batch_idx(
                    data=y):
                net = self._net_input(X[idx], self.w_, self.b_)
                softm = self._softmax(net)
                diff = softm - y_enc[idx]

                grad = np.dot(X[idx].T, diff)
                
                self.w_ -= (self.learning_rate * grad +
                            self.learning_rate * self.lamda * self.w_)
                self.b_ -= (self.learning_rate * np.sum(diff, axis=0))

            net = self._net_input(X, self.w_, self.b_)
            softm = self._softmax(net)
            cross_ent = self._cross_entropy(output=softm, y_target=y_enc)
            cost = self._cost(cross_ent)
            self.cost_.append(cost)
        return self

    def train(self, X, y, init_params=True):
        self._train(X=X, y=y, init_params=init_params)
        self._is_fitted = True
        return self
    
    def _predict(self, X):
        probs = self.predict_prob(X)
        return self._to_classlabels(probs)
 
    def predict(self, X):
        return self._predict(X)

    def predict_prob(self, X):
        net = self._net_input(X, self.w_, self.b_)
        softm = self._softmax(net)
        return softm

    def _net_input(self, X, W, b):
        return (X.dot(W) + b)

    def _softmax(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum()

    def _cross_entropy(self, output, y_target):
        epsilon = 1e-5
        return - np.sum(np.log(output + epsilon) * (y_target), axis=1)

    def _cost(self, cross_entropy):
        L1_term = self.lamda * np.sum(np.abs(self.w_))
        L2_term = self.lamda * np.sum(self.w_ ** 2)
        if self.regulariser == 'l1':
            cross_entropy = cross_entropy + L1_term
        if self.regulariser == 'l2':
            cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def _to_classlabels(self, z):
        return z.argmax(axis=1)
    
    def _init_params(self, weights_shape, bias_shape=(1,), dtype='float64',
                     scale=1):
        w = np.random.normal(loc=0.0, scale=scale, size=weights_shape)
        b = np.zeros(shape=bias_shape)
        return b.astype(dtype), w.astype(dtype)
    
    def _one_hot(self, y, n_labels, dtype):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)    
    
    def _yield_batch_idx(self, data):
        indices = np.arange(data.shape[0])
        yield indices