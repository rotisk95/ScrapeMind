import torch
from bindsnet.network.nodes import LIFNodes, Nodes

class RUMNodes(LIFNodes):
    def __init__(self, n, dt, alpha=0.95, beta=0.8, **kwargs):
        super().__init__(n, **kwargs)  # Call the __init__ method of LIFNodes

        # Initialize additional state variables for the RUM dynamics
        self.phase = torch.rand(n) * 2 * torch.pi  # Initialize to random values between 0 and 2*pi
        self.amplitude = torch.rand(n)  # Initialize to random values between 0 and 1

        # Initialize additional parameters for the RUM dynamics
        self.tau_phi = torch.tensor(0.5)  # Slower dynamics
        self.w = torch.randn(n, n) * 0.01  # Small random values for weight initialization
        # Set baseline current with small random fluctuations around a mean value.
        baseline_current = 0.1  # This is an arbitrary baseline value; you may want to tune it.
        random_fluctuation = 0.012  # Small random fluctuation; tune as needed.
        
        self.I = torch.full((n,), baseline_current) + torch.randn(n) * random_fluctuation
        
        self.tau_A = torch.tensor(0.5)  # Slower dynamics for amplitude
        self.alpha = torch.tensor(alpha, dtype=torch.float)  # Scaling factor for phase
        self.beta = torch.tensor(beta, dtype=torch.float)  # Scaling factor for amplitude

        # Add the dt value
        self.dt = dt

    def calculate_phase_change(self):
        tau_phi = self.tau_phi 
        w = self.w
        I = self.I
        phi = self.phase
        A = self.amplitude
        phase_change = (torch.pi / 2 + torch.matmul(w, A * torch.sin(phi)) + I) / tau_phi 
        phase_change *= self.alpha  # Apply scaling
        return phase_change

    def calculate_amplitude_change(self):
        tau_A = self.tau_A
        phi = self.phase
        A = self.amplitude
        amplitude_change = (-A + (1 + torch.cos(phi)) / 2) / tau_A
        amplitude_change *= self.beta  # Apply scaling
        return amplitude_change

    def forward(self, x=None):
        super().forward(x)  # Call the parent class's forward method
        self.phase += self.dt * self.calculate_phase_change()
        self.amplitude += self.dt * self.calculate_amplitude_change()
