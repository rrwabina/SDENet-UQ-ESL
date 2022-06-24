import numpy as np
import torch 
from scipy import stats 

'''
Acquisition functions appropriate for image data settings
Args: 
        learner: model that measures uncertainty after training
        X: pool set to select uncertainty
        n_instances: number of points that randomly select from pool set
'''

def predictions_from_pool(model, X_pool: np.ndarray, T: int = 100, training: bool = True):
    random_subset = np.random.choice(range(len(X_pool)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(model.estimator.forward(X_pool[random_subset], 
                                training=training), dim = -1).cpu().numpy() for _ in range(T)])
    return outputs, random_subset

def uniform(learner, X, n_instances = 10):
    '''
    Baseline AF returning a draw from a uniform distribution over [0, 1]
    '''
    query_idx = np.random.choice(range(len(X)), size = n_instances, replace = False)
    return query_idx, X[query_idx]

def shannon_entropy(learner, X, n_instances = 1, T = 100, shannon: bool = False):
    random_subset = np.random.choice(range(len(X)), size = 25, replace=False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], 
                                training = True), dim = -1).cpu().numpy() for _ in range(T)])
    pc = outputs.mean(axis=0)
    H = (-pc * np.log(pc + 1e-10)).sum(axis = -1)  
    if shannon:
        E = -np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)
        return H, E, random_subset
    return H, random_subset

def max_entropy(learner, X, n_instances = 1, T = 100):
    '''
    Max Entropy: choose pool points that maximize predictive entropy
        (Shannon, 1948)
    '''
    random_subset = np.random.choice(range(len(X)), size = 25, replace = False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True), dim = -1).cpu().numpy()
                            for t in range(100)])
    pc = outputs.mean(axis = 0)
    acquisition = (-pc * np.log(pc + 1e-10)).sum(axis = -1)
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]

def bald(learner, X, n_instances = 1, T = 100):
    '''
    Bayesian Active Learning by Disagreement
        Choose pool points that are expected to maximize the information gained
        about the model parameters, i.e., maximize the mutual information between
        the predictions and model posterior (Houlsby et al., 2011)
    '''
    random_subset = np.random.choice(range(len(X)), size = 25, replace = False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True),dim=-1).cpu().numpy()
                            for t in range(100)])
    pc = outputs.mean(axis=0)
    H = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    E_H = - np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)  
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]   

def variations_ratios(learner, X, n_instances = 1, T = 100):
    '''
    Variation Ratios measures the lack of confidence
    '''
    random_subset = np.random.choice(range(len(X)), size = 25, replace = False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True),dim=-1).cpu().numpy()
                            for t in range(100)])
    predictions = np.argmax(outputs, axis = 2)
    _, count = stats.mode(predictions, axis = 0)
    acquisition = (1 - count / predictions.shape[1]).reshape((-1))
    idx = (- acquisition).argsort()[: n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]

def mean_std(learner, X, n_instances = 1, T = 100):
    '''
    Maximize the mean and standard deviation
    '''
    random_subset = np.random.choice(range(len(X)), size = 25, replace = False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(X[random_subset], training=True),dim=-1).cpu().numpy()
                            for t in range(100)])
    sigma_c = np.std(outputs, axis = 0)
    acquisition = np.mean(sigma_c, axis = -1)
    idx = (- acquisition).argsort()[: n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]
 