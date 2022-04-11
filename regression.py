import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the prior distribution
    
    Outputs: None
    -----
    """

    mu = np.array([0., 0.])
    cov = np.array([[beta, 0.],[0., beta]])

    N = 100
    X = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(X, Y)
    x = X.ravel()
    y = Y.ravel()

    density = util.density_Gaussian(mu, cov, np.column_stack((x, y)))

    plt.figure(1)
    plt.scatter(-0.1, -0.5, marker='o', c='red', s=70)
    plt.contour(X, Y, density.reshape(X.shape))
    plt.title('Prior')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.show()

    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """

    #prior
    mu_prior = np.array([0., 0.])
    cov_prior = np.array([[beta, 0.],[0., beta]])

    #design matrix choice can be simply adding 1 as phi_0
    phi = np.ones((x.size, 2))
    phi[:,1] = x.ravel()

    #as stated in textbook
    Cov = np.linalg.inv(np.linalg.inv(cov_prior) + 1/sigma2*np.dot(np.transpose(phi),phi))
    mu = np.dot(Cov, 1/sigma2*np.dot(np.transpose(phi),z)).ravel()

    N = 100
    W = np.linspace(-1, 1, N)
    Y = np.linspace(-1, 1, N)
    W, Y = np.meshgrid(W, Y)
    w = W.ravel()
    y = Y.ravel()

    density = util.density_Gaussian(mu, Cov, np.column_stack((w, y)))

    plt.figure(2)
    plt.contour(W, Y, density.reshape(W.shape))
    plt.scatter(-0.1, -0.5, marker='o', c='red', s=70)
    plt.title('Posterior 5')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.show()

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """

    x = np.array(x)

    phi = np.ones((x.size, 2))
    phi[:,1] = x.ravel()

    #z = a1*x + a0 + w
    predictive_mu = np.dot(phi, mu) #a1*x + a0
    predictive_cov = sigma2 + np.dot(phi, np.dot(Cov,phi.T)) #w error term

    predictive_var = np.sqrt(predictive_cov.diagonal())

    z_hat = predictive_mu

    plt.figure(3)
    plt.ylim(-4.0, 4.0)
    plt.xlim(-4.0, 4.0)

    for i in range(0, x_train.size):
        plt.scatter(x_train[i], z_train[i], marker='o', c='black', s=15)
    plt.errorbar(x, z_hat, yerr=predictive_var)

    plt.title('Predictive 5')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()

    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns = 5 #we want for 1, 5, and 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
