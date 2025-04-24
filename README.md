# 🧬 Urban Infection Simulation using OSMnx — Toyosu Crowd Diffusion Model

This project simulates infectious disease spread and crowd movement in an urban road network using OpenStreetMap data (via OSMnx) and a custom agent-based simulation framework. Each individual's trajectory is tracked in time and space.

## 📌 Project Overview

- **Area**: Toyosu District, Koto-ku, Tokyo  
- **Network Type**: Pedestrian (`walk`) graph from OpenStreetMap  
- **Graph Size**: 990 nodes, 2972 edges  
- **Goal**: To model both human flow and infection propagation in a spatial network using mathematically grounded simulation logic.

## 🧠 Key Feature: Trajectory-Based Agent Modeling

The core design is a trajectory-tracking system that models each individual separately. The simulation records agents in a 3D array structure:

```
X1: (T, N, M) - uninfected individuals  
X2: (T, N, M) - infected individuals
```

- `T`: time steps  
- `N`: number of nodes (990)  
- `M`: individuals per node (10)  

✅ This structure allows:  
- Full trajectory tracing  
- Infection state monitoring over time  
- Spatial aggregation and visualization  
- High compatibility with graph neural network (GNN) input formats

## 📐 Mathematical Modeling

### 🧭 Human Movement (Markov Process)

Movement between nodes is modeled using a stochastic transition matrix `PA`:

- Constructed by element-wise multiplying the adjacency matrix `A` with a random matrix `T`, then normalizing each row.  
- Resulting matrix `PA` ∈ ℝ^(990×990), where all row sums = 1.

**Update Rule**:

```
X_{t+1} = PA^T · X_t
```

This represents agent movement across the graph via a discrete-time Markov process.

### 🔄 Permutation Invariance

A key theoretical feature of the model is **permutation invariance**:

```
PA · X = X
```

This means the simulation result is **invariant under any reordering of individuals**. The model captures **spatial transitions only**, making it robust and generalizable for large-scale crowd dynamics. This is a mathematically rigorous property that ensures fairness and structural clarity in the agent-based flow model.

### 🦠 Infection Dynamics (SI Model)

The infection model follows the Susceptible-Infected (SI) structure:

```
dX1/dt = -β * (X1 * X2) / (X1 + X2)  
dX2/dt =  β * (X1 * X2) / (X1 + X2)
```

- `X1`: susceptible population  
- `X2`: infected population  
- `β`: infection rate (can be constant or sampled per time step)  

This formulation models **infection pressure within each node** and is applied independently per node. Numerical integration is performed via **Euler's method**.

## 🧪 Simulation Flow

1. **Initialization**:  
   - Place 10 individuals at every node (uninfected, X1)  
   - Infect 5 individuals at a single randomly selected node (X2)

2. **Time Evolution (repeated T steps)**:  
   - Each individual randomly walks to one of the neighboring nodes  
   - Infection is evaluated locally at each node using the SI formula  
   - All states are recorded in `X1` and `X2` trajectory arrays

## 🖼️ Visualization

The simulation includes visual output of both infection spread and agent movement.

- **Blue**: uninfected individuals  
- **Red**: infected individuals  
- **Green**: network nodes  
- Agent paths are plotted over time to illustrate flow and spread  
- Optional: animate simulation using `matplotlib.animation.FuncAnimation`
