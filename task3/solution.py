"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel # linear
from scipy.stats import norminvgauss, norm



# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
LAGR_LAMBDA = 4
EXC_INVALID = True

#kernel_f = Matern(length_scale_bounds=(1e-05, 100000.0), nu=2.5)
#kernel_v = DotProduct() + Matern(length_scale_bounds=(1e-05, 100000.0), nu=2.5)

pot_len_scales = [0.5, 1, 10]
## Tried combos: (0,0), (1,1), (2,2), (0,2), (2,0),                                                                                           ##( (0,1), (1,0)
## Best combo:   (2,0) -> 0.644                  (0,0) (->0.629)              (0,2) -> 0.621
kernel_f = Matern(length_scale=pot_len_scales[2], length_scale_bounds=(1e-05, 100000.0), nu=2.5)
kernel_v = Matern(length_scale=pot_len_scales[1], length_scale_bounds=(1e-05, 100000.0), nu=2.5) + DotProduct() +WhiteKernel(noise_level=1e-5)




# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # K: Empty list to fill with datapoints
        self.x = []
        self.v = []
        self.f = []
        # bioavailability (logP) in [0,1]
        # Minimal synthetic acessiblity score (SA) --> high SA more difficult to synthesize
        # self.x_init = 0 # initial point in domain
        
        self.gauss_pr_f = GaussianProcessRegressor(kernel = kernel_f, alpha = 0.15) # target function? OR RBF kernel, add noise?
        self.gauss_pr_v = GaussianProcessRegressor(kernel = kernel_v, alpha = 1e-4)


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        # return self.optimize_acquisition_function()
        
        if EXC_INVALID:  
            while True:
                x_opt = self.optimize_acquisition_function()
                v_mean, v_std = self.gauss_pr_v.predict([[x_opt]], return_std=True)
                if (v_mean < SAFETY_THRESHOLD):
                    return x_opt
        else:
            return self.optimize_acquisition_function()
                   
    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt


    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)
 
        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        
        # TODO: Implement the acquisition function you want to optimize.
        # UCB
        f_mean, f_std = self.gauss_pr_f.predict(x, return_std=True)
        f             = f_mean + BETA*f_std

        v_mean, v_std = self.gauss_pr_v.predict(x, return_std=True)
        v = v_mean + BETA*v_std

        # PI
        # f_opt = max(self.f)
        # f = norm.cdf((f_mean-f_opt)/f_std)

        return f - LAGR_LAMBDA * max(v, 0)

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.x.append(x)
        self.f.append(f)
        self.v.append(v)

        # Update the model with the new data
        xs = np.reshape(self.x, (-1, 1))
        self.gauss_pr_f.fit(xs, self.f)
        self.gauss_pr_v.fit(xs, self.v)
    

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        ## TODO: Return your predicted safe optimum of f.
        #fs = np.asarray(self.f)
        #idx = np.argmax(fs)
        #return self.x[idx]


        # Initialize variables to track the best point and its corresponding value
        best_x = None
        max_f = float('-inf')  # Start with the lowest possible value

        # Iterate through each set of points
        for i in range(len(self.x)):
            if self.v[i] < SAFETY_THRESHOLD and self.f[i] > max_f:
                max_f = self.f[i]
                best_x = self.x[i]
        return best_x


    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float): # object value f(x), matern
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float): # cost value SA v(x)
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
