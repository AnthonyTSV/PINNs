import os
import torch

# Use PyTorch backend for DeepXDE
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SteadyStateHeat2DPINN:
    """
    Solve steady-state heat equation (Laplace) in 2D:
        alpha * (d^2T/dx^2 + d^2T/dy^2) = 0
    on a rectangular domain [0, width] x [0, length].
    
    """

    def __init__(
        self,
        # Geometry + PDE parameters
        width=1.0,           # domain in x-direction: [0, width]
        length=1.0,          # domain in y-direction: [0, length]
        alpha=1.0,           # diffusion coefficient
        # PINN/Training hyperparameters
        sample_points=2000,
        architecture=(2,) + (60,)*5 + (1,),
        activation="tanh",
        initializer="Glorot uniform",
        learning_rate=1e-3,
        loss_weights=(1, 1, 1, 1, 1),
        iterations=10000,
        optimizer="adam",
        batch_size=32,
        # Device
        device=None
    ):
        """
        Constructor initializes geometry, PDE parameters, network architecture, and other hyperparameters.
        """
        # PDE / domain
        self.width = width
        self.length = length
        self.alpha = alpha

        # Training / model hyperparameters
        self.sample_points = sample_points
        self.architecture = architecture
        self.activation = activation
        self.initializer = initializer
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.iterations = iterations
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        torch.device(self.device)

        # Internal placeholders for DeepXDE geometry, data, net, and model
        self.geom = None
        self.data = None
        self.net = None
        self.model = None

    # ----------------------------------------------------------------------
    # 1. PDE and Boundary Definitions
    # ----------------------------------------------------------------------

    def pde(self, X, T):
        """
        Steady-state PDE: alpha*(d^2 T/dx^2 + d^2 T/dy^2) = 0
        """
        dT_xx = dde.grad.hessian(T, X, i=0, j=0)
        dT_yy = dde.grad.hessian(T, X, i=1, j=1)
        return self.alpha * (dT_xx + dT_yy)  # = 0

    @staticmethod
    def boundary_left(X, on_boundary):
        """
        x = 0
        """
        x, _ = X
        return on_boundary and np.isclose(x, 0.0)

    @staticmethod
    def boundary_right(X, on_boundary):
        """
        x = width
        """
        x, _ = X
        return on_boundary and np.isclose(x, 1.0)

    @staticmethod
    def boundary_top(X, on_boundary):
        """
        y = length
        """
        _, y = X
        return on_boundary and np.isclose(y, 1.0)

    @staticmethod
    def boundary_bottom(X, on_boundary):
        """
        y = 0
        """
        _, y = X
        return on_boundary and np.isclose(y, 0.0)

    @staticmethod
    def dirichlet_left_val(X):
        """
        Dirichlet BC on the left: T = 1
        """
        return np.full((len(X), 1), 25)

    @staticmethod
    def dirichlet_right_val(X):
        """
        Dirichlet BC on the right: T = 0
        """
        return np.full((len(X), 1), 5)

    @staticmethod
    def zero_flux_val(X):
        """
        For Neumann BC: dT/dn = 0 => 0
        """
        return np.zeros((len(X), 1))

    # ----------------------------------------------------------------------
    # 2. Geometry, Data Setup
    # ----------------------------------------------------------------------

    def build_geometry(self):
        """
        Define the 2D rectangular geometry for [0, width] x [0, length].
        """
        self.geom = dde.geometry.Rectangle([0.0, 0.0], [self.width, self.length])

    def build_data(self):
        """
        Create the PDE data object with PDE, BCs, no time dimension.
        - Dirichlet at x=0 (T=1) and x=width (T=1)
        - Neumann (zero flux) at y=0, y=length
        """
        bc_left = dde.DirichletBC(
            self.geom,
            self.dirichlet_left_val,
            self.boundary_left
        )
        bc_right = dde.DirichletBC(
            self.geom,
            self.dirichlet_right_val,
            self.boundary_right
        )
        bc_top = dde.DirichletBC(
            self.geom,
            self.dirichlet_right_val,
            self.boundary_top
        )
        bc_bottom = dde.DirichletBC(
            self.geom,
            self.dirichlet_right_val,
            self.boundary_bottom
        )

        self.data = dde.data.PDE(
            geometry=self.geom,
            pde=self.pde,
            bcs=[bc_left, bc_top, bc_right, bc_bottom],
            num_domain=int(self.sample_points),
            num_boundary=int(self.sample_points / 4),
        )

    # ----------------------------------------------------------------------
    # 3. Model: Neural Network Definition and Compilation
    # ----------------------------------------------------------------------

    def build_model(self):
        """
        Build the neural network, define Model, and compile.
        """
        self.net = dde.maps.FNN(self.architecture, self.activation, self.initializer)
        # Example: apply an output transform if desired
        # self.net.apply_output_transform(lambda _, y: abs(y))

        self.model = dde.Model(self.data, self.net)

        # Compile the model
        self.model.compile(
            self.optimizer,
            lr=self.learning_rate,
            loss_weights=self.loss_weights,
        )

    # ----------------------------------------------------------------------
    # 4. Training & Adaptive Refinement
    # ----------------------------------------------------------------------

    def train_model(self, iterations=None):
        """
        Train the model for the specified number of iterations.
        """
        if iterations is None:
            iterations = self.iterations

        early_stopping = dde.callbacks.EarlyStopping(
            patience=5000,
            min_delta=1e-5,
        )

        losshistory, train_state = self.model.train(
            iterations=iterations,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
        )
        self.model.compile("L-BFGS")
        losshistory, train_state = self.model.train(
            batch_size=self.batch_size,
            callbacks=[early_stopping],
        )

        return losshistory, train_state

    # ----------------------------------------------------------------------
    # 5. Saving & Plotting
    # ----------------------------------------------------------------------

    def save_model(self, filepath="./trained_PINN_model"):
        """
        Save the trained model parameters.
        """
        if self.model is not None:
            self.model.save(filepath)

    def plot_best_state(self, train_state, filename="best_state.pdf"):
        """
        Plot the best state (lowest loss) during training.
        """
        plt.figure()
        dde.utils.plot_best_state(train_state)
        plt.savefig(filename, dpi=300)
        print(f"Best state plot saved to {filename}")

    def plot_loss_history(self, losshistory, filename="loss_history.pdf"):
        """
        Plot and save the training loss history.
        """
        plt.figure()
        dde.utils.plot_loss_history(losshistory)
        plt.savefig(filename, dpi=300)

    def plot_residual(self, model, filename="residual.pdf"):
        """
        Plot the residual of the trained model.
        """
        x = self.geom.random_points(1000)
        f_pred = model.predict(x, operator=self.pde)
        f_pred = f_pred.reshape(-1, 1)
        plt.figure()
        contour = plt.imshow(
            f_pred,
            origin="lower",
            extent=[0, self.width, 0, self.length],
            interpolation="bilinear",
            cmap="jet",
            aspect="auto",
        )
        plt.colorbar(contour, label="Temperature")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("PDE Residual")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Residual plot saved to {filename}")

    # ----------------------------------------------------------------------
    # 6. Visualization (Static)
    # ----------------------------------------------------------------------

    def visualize_solution(self, model, nx=100, ny=100, filename_solution="solution.pdf"):
        """
        Evaluate the PINN solution on a 2D grid, and create a static contour or heatmap.
        """
        x_coords = np.linspace(0, self.width, nx)
        y_coords = np.linspace(0, self.length, ny)
        X, Y = np.meshgrid(x_coords, y_coords)
        grid_points = np.vstack((X.ravel(), Y.ravel())).T

        # Predict solution
        T_pred = model.predict(grid_points)
        T_pred = T_pred.reshape((ny, nx))

        # Plot
        plt.figure()
        contour = plt.imshow(
            T_pred,
            origin="lower",
            extent=[0, self.width, 0, self.length],
            interpolation="bilinear",
            cmap="jet",
            aspect="auto"
        )
        plt.colorbar(contour, label="Temperature")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Steady-State Temperature Distribution")
        plt.savefig(filename_solution, dpi=300)
        plt.close()
        print(f"Solution plot saved to {filename_solution}")

    # ----------------------------------------------------------------------
    # 7. Orchestrator Method
    # ----------------------------------------------------------------------

    def run(self):
        """
        High-level method to orchestrate the entire process:
        """
        self.build_geometry()
        self.build_data()

        self.build_model()

        losshistory, train_state = self.train_model()

        # self.save_model()

        filename_loss_history = os.path.join(BASE_DIR, "loss_history.pdf")
        self.plot_loss_history(losshistory, filename=filename_loss_history)

        filename_best_state = os.path.join(BASE_DIR, "best_state.pdf")
        self.plot_best_state(train_state, filename=filename_best_state)

        filename_residual = os.path.join(BASE_DIR, "residual.pdf")
        self.plot_residual(self.model, filename=filename_residual)

        filename_solution = os.path.join(BASE_DIR, "solution.pdf")
        self.visualize_solution(self.model, filename_solution=filename_solution)


if __name__ == "__main__":
    solver = SteadyStateHeat2DPINN(
        width=1.0,
        length=1.0,
        alpha=1.0,
        sample_points=2000,
        architecture=(2,) + (50,)*4 + (1,),
        activation="tanh",
        initializer="Glorot uniform",
        learning_rate=1e-3,
        loss_weights=(1, 1, 1, 1, 1),
        iterations=10000,
        optimizer="adam",
        batch_size=16,
    )
    solver.run()
