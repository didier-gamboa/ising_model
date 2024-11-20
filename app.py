import numpy as np
import pandas as pd
import streamlit as st
from src import randomlattice, simulate_metropolis
import matplotlib.pyplot as plt

st.title("Olimpiada Nacional de Física")

# Parámetros de simulación
st.sidebar.title("Pámetros de la Simulación")
L = st.sidebar.number_input("Tamaño de la red (L)", min_value=10, max_value=100, value=20)
temperatura = st.sidebar.slider("Temperatura (T)", min_value=0.1, max_value=5.0, value=2.0)
ciclos = st.sidebar.number_input("Número de ciclos de Monte Carlo", min_value=10, max_value=750, value=50)


if st.sidebar.button("Simular"):
    N_samples = 10

    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    sample_magnetisation = []

    for i in range(N_samples):

        
        lattice = randomlattice(L, L)
        initial_lattice = lattice.copy()

        magnetisation, final_lattice = simulate_metropolis(ciclos, lattice, temperatura)
        sample_magnetisation.append(np.mean(abs(magnetisation)))

        progress_bar.progress((i + 1) / N_samples)
        progress_text.text(f"Progreso: {i + 1} de {N_samples} experimentos completados")

    st.header("Red Inicial")
    fig1, ax1 = plt.subplots(figsize=(1, 1))  
    ax1.imshow(initial_lattice, cmap='binary', interpolation='nearest')
    ax1.axis('off') 
    st.pyplot(fig1)

    st.header("Red Final")
    fig2, ax2 = plt.subplots(figsize=(1, 1))  
    ax2.imshow(final_lattice, cmap='binary', interpolation='nearest')
    ax2.axis('off') 
    st.pyplot(fig2)

    data = pd.DataFrame(
        [sample_magnetisation],  # Fila con valores
    )
    data.index = ["Magnetización"]  

    mean_magnetisation = np.mean(sample_magnetisation)

    st.header("Magnetización en cada Experimento")
    fig3, ax3 = plt.subplots(figsize=(6, 4))  
    ax3.scatter(range(1, 11), sample_magnetisation, color='dodgerblue', marker='o', s=50)
    ax3.axhline(mean_magnetisation, color='red', linestyle='--', label=f'Promedio: {mean_magnetisation:.2f}')
    ax3.set_xlabel('Experimento')
    ax3.set_ylabel('Magnetización')
    ax3.set_title('Magnetización en cada Experimento')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig3)

    st.table(data)