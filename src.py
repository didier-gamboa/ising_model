import numpy as np
import matplotlib.pyplot as plt

def randomlattice(N, M):
    """
    
    Generates a random lattice with dimension NxM filled with spins randomly set to either +1 or -1.
    
    Parameters:
        N (int): Number of rows in the lattice.
        M (int): Number of columns in the lattice.
        
    Returns:
        ndarray: A 2D array representing the generated lattice
    """
    lattice = np.zeros((N, M), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            lattice[i, j] = 1 if np.random.rand() < 0.5 else -1
    return lattice

def plotlattice(lattice):
    """
    Plots the given lattice using matplotlib imshow function.

    Parameters:
        lattice (ndarray): A 2D array representing the lattice to be plotted.
    Outputs: 
        Visualised array using binary cmap, +1 (Black), -1 (White).
    """
    plt.imshow(lattice, cmap='binary', interpolation='nearest')
    plt.show()

def combined_spin_func(lattice,i,j):
    """
    Calculates the combined spin value for each lattice site by summing over the spins of its neighboring sites.

    Parameters:
        lattice (ndarray): A 2D array representing the lattice.

    Returns:
        ndarray: 1D array containing the combined spin value for each lattice site.
    """
    N,M = lattice.shape
    combined_spin = (
            lattice[i - 1, j] +  # Top neighbor
            lattice[(i + 1) % N, j] +  # Bottom neighbor
            lattice[i, j - 1] +  # Left neighbor
            lattice[i, (j + 1) % M]  # Right neighbor
    )
    return combined_spin


def calculate_dE(lattice, i, j):
    """
    Calculates the change in energy (dE) for a spin flip at a given lattice site.

    Parameters:
        lattice (ndarray): A 2D array representing the lattice.
        i (int): Row index of the lattice site.
        j (int): Column index of the lattice site.

    Returns:
        int: Change in energy (dE) due to the spin flip.
    """
    current_spin = lattice[i, j]
    combined_spin = combined_spin_func(lattice, i, j)
    dE = 2 * current_spin * combined_spin
    return dE

def calculate_energy(lattice, N, M):
    """
    Calculates the total energy of the lattice.

    Parameters:
        lattice (ndarray): A 2D array representing the lattice.
        N (int): Number of rows in the lattice.
        M (int): Number of columns in the lattice.

    Returns:
        float: Total energy of the lattice.
    """
    energy = 0
    for i in range(N):
        for j in range(M):
            energy += calculate_dE(lattice, i, j)
    return energy / 2  # Dividing by 2 to avoid double-counting interactions

def metropolis(lattice, T_prime):
    """
    Performs the Metropolis algorithm for a given lattice at a given temperature.

    Parameters:
        lattice (ndarray): A 2D array representing the lattice.
        T_prime (float): The temperature parameter.

    Returns:
        ndarray: The updated lattice after applying the Metropolis algorithm.
    """
    N, M = lattice.shape
    random_indices = (np.random.rand(N * M) * (N * M)).astype(np.int64)

    def calculate_dE(lattice, i, j):
        """
        Calculates the change in energy (dE) for a spin flip at a given lattice site.

        Parameters:
            lattice (ndarray): A 2D array representing the lattice.
            i (int): Row index of the lattice site.
            j (int): Column index of the lattice site.

        Returns:
            int: Change in energy (dE) due to the spin flip.
        """

        current_spin = lattice[i, j]
        combined_spin = (
            lattice[i - 1, j] +  # Top neighbor
            lattice[(i + 1) % N, j] +  # Bottom neighbor
            lattice[i, j - 1] +  # Left neighbor
            lattice[i, (j + 1) % M]  # Right neighbor
        )
        dE = 2 * current_spin * combined_spin
        return dE

    for idx in range(N * M):
        flat_index = random_indices[idx]
        i, j = flat_index // M, flat_index % M
        dE = calculate_dE(lattice, i, j)  # Use calculate_dE inline
        probability = np.exp(-dE / T_prime)
        r = np.random.rand()
        if r < probability or dE < 0:
            lattice[i, j] *= -1
    return lattice

def simulate_metropolis(N_mc, lattice, T_prime):
    magnetisation = []
    stable_count = 0
    for i in range(N_mc):
        lattice = metropolis(lattice, T_prime)
        magnetisation.append(np.mean(lattice))
        if i > 0 and magnetisation[i] == magnetisation[i-1]:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= 500:
            if i >= 500:
                break

    last_500_magnetisation = np.array(magnetisation[-500:])

    sample_indices = np.linspace(0, len(last_500_magnetisation) - 1, 10, dtype=int)
    sampled_magnetisation = [last_500_magnetisation[idx] for idx in sample_indices]
    mean_mag = np.mean(sampled_magnetisation)
    return mean_mag, lattice