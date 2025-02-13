import os
os.environ["DDE_BACKEND"] = "pytorch"
import torch
import numpy as np

import deepxde as dde
import matplotlib.pyplot as plt
import matplotlib

try:
    import scienceplots
    plt.style.use("science")
except ImportError:
    print("SciencePlots is not available. Using default style.")
    plt.style.use("default")

matplotlib.rcParams["font.size"] = "16"
plt.rcParams["figure.figsize"] = (8, 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.device(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class HeatEquation1DPINN:
    """PINN for the 1D steady-state heat equation:
         -k d^2T/dx^2 = Qg,   0 <= x <= 1,
         with T(0)=0, T(1)=0.
    """

    def __init__(self, k_value=0.5, Qg_value=10.0, num_domain=30, num_boundary=2):
        """
        k_value:   Thermal conductivity (W/m.K)
        Qg_value:  Volumetric heat source (W/m^3)
        num_domain, num_boundary: Number of collocation points and boundary points
        """
        self.k = k_value
        self.Qg = Qg_value
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.activation_func = "tanh"

        # Build PDE data and network model
        self.data = self._build_data()
        self.net = self._build_network()
        self.model = dde.Model(self.data, self.net)

    def heat_equation_residual(self, x, y):
        """
        PDE residual: T''(x) + Qg/k = 0
        """
        d2T_dx2 = dde.grad.hessian(y, x, i=0, j=0)
        return d2T_dx2 + self.Qg / self.k

    def heat_exact(self, x):
        """Analytical solution for checking:
           T_exact(x) = (Qg / (2k)) * x * (1 - x).
        """
        return (self.Qg / (2 * self.k)) * x * (1.0 - x)

    @staticmethod
    def boundary_left(x, on_boundary):
        # x is a NumPy array [x_value].
        return on_boundary and np.isclose(x[0], 0.0)

    @staticmethod
    def boundary_right(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1.0)

    def _build_data(self):
        """Construct the DeepXDE PDE data object (geometry, PDE, BCs)."""
        geom = dde.geometry.Interval(0.0, 1.0)

        bc_left = dde.DirichletBC(
            geom,
            lambda x: 0.0,  # T(0) = 0
            self.boundary_left
        )
        bc_right = dde.DirichletBC(
            geom,
            lambda x: 0.0,  # T(1) = 0
            self.boundary_right
        )

        data = dde.data.PDE(
            geometry=geom,
            pde=self.heat_equation_residual,
            bcs=[bc_left, bc_right],
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
            solution=self.heat_exact,
        )
        return data

    def _build_network(self):
        """Construct the neural network (FNN)."""
        layer_size = [1] + [30] * 3 + [1]  # [input_dim] + hidden_layers + [output_dim]
        initializer = "Glorot uniform"
        return dde.nn.FNN(layer_size, self.activation_func, initializer)

    def train_model(self, adam_iterations=5000, lr=1e-4):
        """Train the PINN with Adam, then refine with L-BFGS."""
        torch.cuda.empty_cache()
        self.model.compile("adam", lr=lr)
        losshistory, train_state = self.model.train(iterations=adam_iterations)

        # Fine-tune with L-BFGS
        self.model.compile("L-BFGS")
        losshistory, train_state = self.model.train()

        return losshistory, train_state

    def predict(self, x):
        """Evaluate the trained model at given points x (NumPy array)."""
        return self.model.predict(x)

    def plot_solution(self, filename=None):
        """Plot PINN prediction vs. analytical solution."""
        x_test = np.linspace(0, 1, 100)[:, None]
        y_pred = self.predict(x_test)
        y_exact = self.heat_exact(x_test)

        plt.figure()
        plt.plot(x_test, y_pred, label="PINN prediction")
        plt.plot(x_test, y_exact, label="Analytical solution", linestyle="--")
        plt.xlabel("X, m")
        plt.ylabel("T, C")
        plt.legend()

        if filename is not None:
            plt.savefig(filename, dpi=300)

    def plot_loss(self, loss_history, filename=None):
        """Plot the loss history."""
        loss_train = np.array([np.sum(loss) for loss in loss_history.loss_train])
        loss_test = np.array([np.sum(loss) for loss in loss_history.loss_test])
        plt.figure()
        plt.semilogy(loss_history.steps, loss_train, label="Train loss")
        plt.semilogy(loss_history.steps, loss_test, label="Test loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()

        if filename is not None:
            plt.savefig(filename, dpi=300)

def test_one_lr(lr):
    print(f"Training with lr={lr}")
    pinn = HeatEquation1DPINN(k_value=0.5, Qg_value=10.0, num_domain=30, num_boundary=2)
    loss_history, train_state = pinn.train_model(adam_iterations=10000, lr=lr)
    x_test = np.linspace(0, 1, 100)[:, None]
    y_pred = pinn.predict(x_test)
    y_true = pinn.heat_exact(x_test)
    f = pinn.model.predict(x_test, operator=pinn.heat_equation_residual)
    mean_residual = np.mean(np.absolute(f))
    l2_error = np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)

    return mean_residual, l2_error, loss_history

def test_lr_range():
    # Train
    mean_residuals = {}
    l2_errors = {}
    lrs = [1e-2, 1e-3, 1e-4, 1e-5]
    loss_histories = {}
    for lr in lrs:
        mean_residual, l2_error, loss_history = test_one_lr(lr)
        mean_residuals[lr] = mean_residual
        l2_errors[lr] = l2_error
        loss_histories[lr] = loss_history
    # print("Learning rates:", lrs)
    # print("Mean residuals:", mean_residuals)
    # print("L2 errors:", l2_errors)

    best_lr = min(l2_errors, key=l2_errors.get)
    print(f"Best learning rate: {best_lr}")

    # plot loss history
    plt.figure()
    for lr in lrs:
        loss_history = loss_histories[lr]
        loss_train = np.array([np.sum(loss) for loss in loss_history.loss_train])
        plt.semilogy(loss_history.steps, loss_train, label=f"lr={lr:.0e}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Best learning rate: {best_lr:.0e}")
    plt.legend()
    all_loss_pdf = os.path.join(BASE_DIR, "all_loss_history.pdf")
    plt.savefig(all_loss_pdf, dpi=300)

def main():
    
    # test_lr_range()
    # Create the PINN object
    pinn = HeatEquation1DPINN(k_value=0.5, Qg_value=10.0, num_domain=30, num_boundary=2)
    # Plot results
    loss_history, train_state = pinn.train_model(adam_iterations=10000, lr=1e-3)
    dde.utils.plot_best_state(train_state)
    plt.savefig(os.path.join(BASE_DIR, "best_state_1d.pdf"), dpi=300)
    x_test = np.linspace(0, 1, 100)[:, None]
    y_pred = pinn.predict(x_test)
    y_true = pinn.heat_exact(x_test)
    f = pinn.model.predict(x_test, operator=pinn.heat_equation_residual)
    mean_residual = np.mean(np.absolute(f))
    l2_error = np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)
    print(f"Mean residual: {mean_residual:.2e}")
    print(f"L2 error: {l2_error:.2e}")

if __name__ == "__main__":
    main()
