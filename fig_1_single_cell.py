import numpy as np
import matplotlib.pyplot as plt

np.random.seed(305) # 11 57 22 19 29 28 53 16 55 49 60 65 74 76 
# 80 89 118 119 145 146 147 191 204 223 226 255 290 297 305
# ============================================================
# TIME
# ============================================================

dt = 0.0034
T = 1000

n = int(T / dt)

t = np.linspace(0, T, n)

# ============================================================
# BASE NOISE
# ============================================================

sigma = 0.4

# ============================================================
# MODEL FUNCTION
# ============================================================

def simulate(A0):

    # --------------------------------------------------------
    # states
    # --------------------------------------------------------

    C = np.zeros(n)
    h = np.zeros(n)
    A = np.ones(n) * A0

    # --------------------------------------------------------
    # initial conditions
    # --------------------------------------------------------

    C[0] = 0.091
    h[0] = 0.8

    # --------------------------------------------------------
    # parameters
    # --------------------------------------------------------

    a = 1.0
    b = 0.8

    # ATP-dependent recovery acceleration
    tau_h_eff = 10.0 / (1 + 0.8 * A0)

    gamma = 0.72

    # --------------------------------------------------------
    # ATP-dependent baseline excitability
    # --------------------------------------------------------

    if A0 < 0.2:
        # near-threshold excitable regime
        I0 = 0.38
    elif A0 < 0.7:
        # transition to oscillatory regime
        I0 = 0.5
    else:
        # strongly driven regime
        I0 = 0.62

    # --------------------------------------------------------
    # ATP-dependent stochasticity
    # higher ATP -> stronger irregularity
    # --------------------------------------------------------

    sigma_eff = sigma * (1 + A0)

    # ========================================================
    # SIMULATION
    # ========================================================

    for i in range(n - 1):

        # stochastic forcing
        noise = (sigma_eff * 3 * np.random.randn())

        # ----------------------------------------------------
        # fast subsystem
        # ----------------------------------------------------

        dC = (C[i] - (C[i]**3) / 3 - h[i] + I0 + gamma * A[i] + noise)

        # ----------------------------------------------------
        # slow recovery
        # ----------------------------------------------------

        dh = (C[i] + a - b * h[i]) / tau_h_eff

        # ----------------------------------------------------
        # update
        # ----------------------------------------------------

        C[i + 1] = C[i] + dt * dC
        h[i + 1] = h[i] + dt * dh

        # ----------------------------------------------------
        # numerical stabilization
        # ----------------------------------------------------

        C[i + 1] = np.clip(C[i + 1],-4,4)

    return C

# ============================================================
# ATP CONDITIONS
# ============================================================

ATP_levels = [0.19, 0.27, 0.9]

labels = [
    "Low ATP (noise-driven excitable regime)",
    "Intermediate ATP (oscillation onset regime)",
    "High ATP (irregular high-frequency regime)"
]

results = []

# ============================================================
# FIGURE 1A–C
# ============================================================

plt.figure(figsize=(12, 7))

for i, A0 in enumerate(ATP_levels):

    C = simulate(A0)

    results.append(C)

    plt.subplot(3, 1, i + 1)
    plt.plot(t,C,linewidth=1.2)
    plt.title(labels[i])
    plt.ylabel("C (Ca$^{2+}$)")
    plt.grid(True, alpha=0.3)

    if i < 2:
        plt.xticks([])

plt.xlabel("Time (a.u.)")
plt.tight_layout()
plt.show()


