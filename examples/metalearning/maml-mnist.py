import autograd.numpy as np
import autograd as ag
from matplotlib import pyplot as plt
import mnist
import pdb


def softmax(X, theta = 1.0, axis = None):
    """
    https://nolanbconaway.github.io/blog/2017/softmax-numpy  
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def relu(z):
    return np.maximum(z, 0.)

def net_predict(params, x):
    """Compute the output of a ReLU MLP with 2 hidden layers."""
    H1 = relu(np.matmul(x, params['W1']) + params['b1'])
    return softmax(np.matmul(H1, params['w2']) + params['b2'],axis=1)
    

def random_init(std, nhid):
    ### YOUR CODE HERE  
    
   
  
 
    ### END CODE 
    
class ToyDataGen:
    """Samples a random piecewise linear function, and then samples noisy
    observations of the function."""
    def __init__(self):

        self.x_train, self.t_train, self.x_test, self.t_test = mnist.load() # loading mnist 
          
        # sorting the training dataset
        inds = self.t_train.argsort()
        self.t_train = self.t_train[inds]
        self.x_train =  self.x_train[inds] 
        self.N_per_class_train,_ = np.histogram((self.t_train))
          
    def sample_dataset(self, npts, idx_start=0, idx_end=5):
        j = np.random.randint(idx_start, idx_end) #  samples digits [0,1,2,3,5] without replacement
        i = j
        while i == j:
          i = np.random.randint(idx_start, idx_end) #  samples digits [0,1,2,3,5] without replacement
             
        range_i = np.arange(np.sum(self.N_per_class_train[0:i]),np.sum(self.N_per_class_train[0:i])+self.N_per_class_train[i],1)
        range_j = np.arange(np.sum(self.N_per_class_train[0:j]),np.sum(self.N_per_class_train[0:j])+self.N_per_class_train[j],1) 
        inds_i = np.random.choice(range_i,size=npts,replace=True) 
        inds_j = np.random.choice(range_j,size=npts,replace=True)
       
        x = self.x_train[np.concatenate((inds_i,inds_j))]
        y = np.concatenate((np.tile([0,1], (npts,1)),np.tile([1,0], (npts,1))))
        x = np.float32(x)
        x -= 0.5
        x /= 0.5

        return x, y,i,j
    
def gd_step(cost, params, lrate):
    """Perform one gradient descent step on the given cost function with learning
    rate lrate. Returns a new set of parameters, and (IMPORTANT) does not modify
    the input parameters."""
    ### YOUR CODE HERE



    ### END CODE


class InnerObjective:
    """Cross Entropy."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __call__(self, params):
        ### YOUR CODE HERE 
        
        ### END CODE
 
class MetaObjective:
    """Cross entropy  after some number of gradient descent steps
    on the inner objective."""
    def __init__(self, x, y, inner_lrate, num_steps,accuracy_fine_tune=False):
        self.x = x
        self.y = y
        self.inner_lrate = inner_lrate
        self.num_steps = num_steps
        self.acc = accuracy_fine_tune 

    def __call__(self, params, return_traj=False):
        """Compute the meta-objective. If return_traj is True, you should return
        a list of the parameters after each update. (This is used for visualization.)"""
        trajectory = [params]

        ### BEGIN STUDENT CODE
        
       
      
     
    
        ### END STUDENT CODE

        if self.acc==True:  
            ### BEGIN STUDENT CODE 
   
  
 

            ### END STUDENT CODE

        if return_traj:
            return final_cost, trajectory
        else:
            return final_cost

    def visualize(self, params, title, ax):
        _, trajectory = self(params, return_traj=True)

        ax.plot(self.x, self.y, 'bx', ms=3.)
        px = np.linspace(XMIN, XMAX, 1000)
        for i, new_params in enumerate(trajectory):
            py = net_predict(new_params, px)
            ax.plot(px, py, 'r-', alpha=(i+1)/len(trajectory))
        ax.set_title(title)


mnist.init()

OUTER_LRATE = 0.0001
OUTER_STEPS = 8000
INNER_LRATE = 0.0005
INNER_STEPS = 1

PRINT_EVERY = 100
DISPLAY_EVERY = 1000

NDATA = 100

INIT_STD = 0.001
NHID = 256

def train():
    np.random.seed(0)
    data_gen = ToyDataGen()
    params = random_init(INIT_STD, NHID)
    fig, ax = plt.subplots(3, 4, figsize=(16, 9))
    plot_id = 0
    
    x_val, y_val,_,_ = data_gen.sample_dataset(NDATA)
    
    for i in range(OUTER_STEPS):
        ### YOUR CODE HERE



        ### END CODE
       
        ### Validation ### 
        if (i+1) % PRINT_EVERY == 0:
            val_cost = MetaObjective(x_val, y_val, INNER_LRATE, INNER_STEPS,accuracy_fine_tune=True) 
            pred = net_predict(params,x_val).argmax(1)
            target = y_val.argmax(1)
            val_acc = 100.*float((pred == target).sum())/float(target.size)
            print('Iteration %d - Meta-objective: %1.3f - Validation Accuracy: %1.2f' % (i+1, val_cost(params), val_acc))
        
    return params     

if __name__ == "__main__":
    final_parameters = train()
    
      

