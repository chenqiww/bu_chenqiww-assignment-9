import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

        # Variables to store activations and gradients
        self.Z1 = None  # Pre-activation of hidden layer
        self.A1 = None  # Activation of hidden layer
        self.Z2 = None  # Pre-activation of output layer
        self.A2 = None  # Output of network (after activation)
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

    def activation(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1  # Linear activation for hidden layer
        self.A1 = self.activation(self.Z1)  # Non-linear activation for hidden layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Linear activation for output layer
        self.A2 = self.sigmoid(self.Z2)  # Sigmoid activation for output layer (binary classification)
        return self.A2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, X, y):
        m = y.shape[0]
        # Compute the derivative of the loss w.r.t output activation
        dA2 = self.A2 - y  # derivative of cross-entropy loss with sigmoid output
        # Compute gradients for W2 and b2
        dW2 = (1 / m) * np.dot(self.A1.T, dA2)
        db2 = (1 / m) * np.sum(dA2, axis=0, keepdims=True)
        # Backpropagate to hidden layer
        dA1 = np.dot(dA2, self.W2.T)
        # Compute derivative of activation function
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        # Compute gradients for W1 and b1
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        # Update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        # Store gradients for visualization
        self.dW1 = dW1
        self.dW2 = dW2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int).reshape(-1, 1)  # Circular boundary
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, fig):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features (Distorted Input Space)
    hidden_features = mlp.A1  # Shape (n_samples, 3)
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)

    # Plot the distorted input space
    # Create a grid in the input space
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 30),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 30))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Project the input grid through the hidden layer
    Z1_grid = np.dot(grid, mlp.W1) + mlp.b1
    A1_grid = mlp.activation(Z1_grid)
    h1_grid = A1_grid[:, 0].reshape(xx.shape)
    h2_grid = A1_grid[:, 1].reshape(xx.shape)
    h3_grid = A1_grid[:, 2].reshape(xx.shape)

    # Plot the surface representing the distorted input space
    ax_hidden.plot_surface(h1_grid, h2_grid, h3_grid, alpha=0.2, color='blue', rstride=1, cstride=1)

    # Plot the decision boundary plane in the hidden space
    # The decision boundary is defined by w[0]*h1 + w[1]*h2 + w[2]*h3 + b = 0
    w = mlp.W2.ravel()
    b = mlp.b2[0, 0]

    # Create a grid for h1 and h2 within the range of hidden features
    h1_range = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 10)
    h2_range = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 10)
    h1_plane, h2_plane = np.meshgrid(h1_range, h2_range)

    # Compute h3 based on the plane equation
    if w[2] != 0:
        h3_plane = (-w[0] * h1_plane - w[1] * h2_plane - b) / w[2]
        # Plot the decision boundary plane
        ax_hidden.plot_surface(h1_plane, h2_plane, h3_plane, alpha=0.2, color='beige')
    else:
        # If w[2] is zero, skip plotting the plane
        pass

    ax_hidden.set_title('Hidden space at step {}'.format(frame * 10))
    ax_hidden.set_xlabel('Neuron 1 activation')
    ax_hidden.set_ylabel('Neuron 2 activation')
    ax_hidden.set_zlabel('Neuron 3 activation')

    # Plot decision boundary in input space
    xx_input, yy_input = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    grid_input = np.c_[xx_input.ravel(), yy_input.ravel()]
    mlp.forward(grid_input)
    A2 = mlp.A2.reshape(xx_input.shape)
    ax_input.contourf(xx_input, yy_input, A2, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title('Input space at step {}'.format(frame * 10))

    # Visualize network and gradients
    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-0.5, 2.5)
    ax_gradient.axis('off')

    # Positions of neurons
    input_layer = [(0, 0.5), (0, 1.5)]
    hidden_layer = [(1, 0), (1, 1), (1, 2)]
    output_layer = [(2, 1)]

    # Draw neurons and add labels
    for idx, (x, y_pos) in enumerate(input_layer):
        circle = Circle((x, y_pos), radius=0.1, fill=True, color='blue')
        ax_gradient.add_patch(circle)
        # Add labels
        ax_gradient.text(x - 0.1, y_pos + 0.2, f'x{idx+1}', ha='center')

    for idx, (x, y_pos) in enumerate(hidden_layer):
        circle = Circle((x, y_pos), radius=0.1, fill=True, color='green')
        ax_gradient.add_patch(circle)
        # Add labels
        ax_gradient.text(x, y_pos + 0.2, f'h{idx+1}', ha='center')

    for idx, (x, y_pos) in enumerate(output_layer):
        circle = Circle((x, y_pos), radius=0.1, fill=True, color='red')
        ax_gradient.add_patch(circle)
        # Add labels
        ax_gradient.text(x + 0.1, y_pos + 0.2, 'y', ha='center')

    # Draw edges with gradient magnitudes
    max_grad = np.max([np.abs(mlp.dW1).max(), np.abs(mlp.dW2).max()])

    # Avoid division by zero
    if max_grad == 0:
        max_grad = 1e-6

    # From input to hidden layer
    for i, (x1, y1) in enumerate(input_layer):
        for j, (x2, y2) in enumerate(hidden_layer):
            grad_mag = np.abs(mlp.dW1[i, j]) / max_grad
            ax_gradient.plot([x1, x2], [y1, y2], linewidth=grad_mag * 5 + 0.1, color='gray')

    # From hidden to output layer
    for i, (x1, y1) in enumerate(hidden_layer):
        x2, y2 = output_layer[0]
        grad_mag = np.abs(mlp.dW2[i, 0]) / max_grad
        ax_gradient.plot([x1, x2], [y1, y2], linewidth=grad_mag * 5 + 0.1, color='gray')

    ax_gradient.set_title('Gradient at step {}'.format(frame * 10))

    # Adjust layout
    fig.tight_layout()

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Ensure that frames are at least 1
    num_frames = max(1, step_num // 10)

    # Set up visualization
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                ax_gradient=ax_gradient, X=X, y=y, fig=fig),
        frames=num_frames,
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
