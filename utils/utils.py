import random
from datetime import datetime

import torch



def f2angle(f):
    """Converts frequency f in [0, 1[ to corresponding angle in [0, 180[."""
    f = f-1/2
    theta = torch.asin(2 * f) + torch.pi/2
    return torch.rad2deg(theta)


def vector_to_toeplitz(v):
    """Converts a vector of size 2N-1 to its corresponding NxN Toeplitz matrix."""
    N = (v.shape[0] + 1) // 2
    r = torch.arange(N)
    c = torch.arange(N)

    diff = r[:, None] - c[None, :]
    idx = diff + N - 1
    t = v[idx]

    return t


def toeplitz_to_vector(X):
    """Converts an NxN Toeplitz matrix to its corresponding vector representation of size 2N-1."""
    N = X.shape[0]
    v = torch.zeros(2 * N - 1, dtype=X.dtype, device=X.device)
    v[:N] = X[0, :].flip(0)
    v[N:] = X[1:, 0]

    return v


def nmse(X1, X2):
    """Computes the normalized mean squared error (NMSE) between two tensors."""
    d = torch.norm(X1 - X2) ** 2
    n = torch.norm(X2) ** 2
    r = torch.mean(d / n)
    return r


def tpm(T, V):
    """Fast Toeplitz matrix-vector product."""
    n, m = V.shape
    c, r = T[:, 0], T[0, :]

    C = torch.cat([c, r[1:].flip(0)])

    Vp = torch.cat(
        [V, torch.zeros(n - 1, m, device=V.device, dtype=V.dtype)], dim=0
    )

    z = torch.fft.ifft(
        torch.fft.fft(C).unsqueeze(1) * torch.fft.fft(Vp, dim=0), dim=0
    )

    return z[:n, :]


def db(x: torch.Tensor):
    """Converts a squared magnitude value to decibels."""
    return 10 * torch.log10(torch.abs(x))


def set_random_seeds(seed: int):
    """Sets the random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_id(network):
    """Generates a unique model identifier based on the model name, number of layers, and timestamp."""
    model_name = network.__class__.__name__.lower()
    num_layers = network.num_layers
    timestamp = datetime.now().strftime("%m%d_%H%M")

    model_tag = f"{model_name}_{num_layers}l_{timestamp}"

    return model_name, model_tag


def randu(shape, a, b):
    """Generates a tensor of dimensions `shape` with samples from a uniform distribution over [a, b]."""
    random_tensor = a + (b - a) * torch.rand(shape)
    return random_tensor


def randn(shape, mean, std_dev):
    """Generates a tensor of dimensions `shape` with samples from a Normal distribution N(mean, std_dev)."""
    random_tensor = mean + std_dev * torch.randn(shape)
    return random_tensor


def randi(low, high, k):
    """
    Returns a tensor of k integers drawn without replacement from the interval [low, high] (inclusive).

    Inputs:
        low (int): Lower bound of the interval.
        high (int): Upper bound of the interval.
        k (int): Number of integers to draw.
    """
    # Ensure the size is not larger than the number of possible values
    assert k <= (high - low + 1), "k cannot be larger than the number of available integers."

    all_integers = torch.arange(low, high + 1)
    random_selection = all_integers[torch.randperm(all_integers.size(0))][:k]

    return random_selection




def rand_freqs (f1, f2, df, k=None):
    """
     Generates k random frequencies within the interval [f1, f2] with a minimum separation df between any two frequencies.
    """
    # If k is not provided, choose a random integer k between 1 and 8
    if k is None:
        k = random.randint(1, 8)

    # Ensure that the interval is large enough to fit k frequencies with minimum distance df
    if (f2 - f1) < (k - 1) * df:
        raise ValueError(f"Interval [{f1}, {f2}] is too small to fit {k} frequencies with minimum distance {df}.")

    # Initialize a tensor to store the k random frequencies
    freqs = torch.zeros(k)

    # Generate the first random frequency within the interval [f1, f2 - (k-1)*df]
    freqs[0] = f1 + random.random() * (f2 - f1 - (k - 1) * df)

    # Generate the remaining k-1 frequencies
    for i in range(1, k):
        # Calculate the maximum possible starting point for the next frequency
        max_start = freqs[i-1] + df

        # Calculate the interval for the next frequency
        remaining_interval = f2 - (k - i) * df

        # Generate the next frequency within the valid interval
        freqs[i] = max_start + random.random() * (remaining_interval - max_start)

    return freqs




def idxf(freqs, frequency_grid):
    """
    Returns the indices of the closest elements in 1D tensor frequency_grid to the elements in 1D tensor freqs.
    """
    K = freqs.size(0)
    N = frequency_grid.size(0)
    closest_indices = torch.zeros(K, dtype=torch.long)  # Initialize the result array

    for i in range(K):
        # Find the position where freqs[i] would fit in frequency_grid
        pos = torch.nonzero(frequency_grid >= freqs[i], as_tuple=True)[0]
        
        # Handle edge cases where pos might be empty
        if pos.numel() == 0:
            closest_indices[i] = N - 1  # Assign the last index of frequency_grid
        elif pos[0] == 0:
            closest_indices[i] = 0  # Assign the first index of frequency_grid
        else:
            # Determine the closest element in frequency_grid
            pos = pos[0]
            if torch.abs(frequency_grid[pos] - freqs[i]) < torch.abs(frequency_grid[pos - 1] - freqs[i]):
                closest_indices[i] = pos
            else:
                closest_indices[i] = pos - 1
    
    closest_indices = torch.sort(closest_indices)[0]
    
    return closest_indices



def k_largest_peaks(x, k):
    """
    Returns the indices of the k largest peaks in a 1D tensor x in order of descending magnitude.
    """
    # Initialize peak mask
    is_peak = torch.zeros(x.shape, dtype=torch.bool)

    # Check interior points
    is_peak[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])

    # Check first and last points
    if len(x) > 1:
        if x[0] > x[1]:
            is_peak[0] = True
        if x[-1] > x[-2]:
            is_peak[-1] = True
    else:
        # If x has only one element, consider it a peak
        is_peak[0] = True

    # Get the values and indices of all peaks
    peak_values = x[is_peak]
    peak_indices = torch.where(is_peak)[0]

    # Sort peaks in descending order
    sorted_indices = torch.argsort(peak_values, descending=True)

    # Get the indices of the k largest peaks
    k = min(k, len(sorted_indices))
    top_k_indices = sorted_indices[:k]

    # Get the original locations of the k largest peaks
    peak_locs = peak_indices[top_k_indices]

    return peak_locs

















