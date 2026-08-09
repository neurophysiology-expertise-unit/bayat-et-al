import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GRID
# ============================================================
Nx, Ny = 10, 10
N = Nx * Ny

# ============================================================
# TIME
# ============================================================
dt = 0.0034
T = 1000
steps = int(T / dt)

alpha_values = np.linspace(0.01, 1.11, 21)

# ============================================================
# MODEL PARAMETERS
# ============================================================
a = 1.0
b = 0.8

sigma = 0.4
eta = 8.0
theta_base = 0.5

target_snaps = [0.175, 0.395, 1.0]

# ============================================================
# LAPLACIAN
# ============================================================
def laplacian(Z):
    return (
        np.roll(Z, 1, axis=0) +
        np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) +
        np.roll(Z, -1, axis=1) -
        4 * Z
    )

# ============================================================
# ACTIVATION FUNCTION
# ============================================================
def Phi(C, theta):
    return 0.5 * (1 + np.tanh(eta * (C - theta)))

# ============================================================
# SIMULATION FUNCTION
# ============================================================
def run_model(disease=False, seed=0):

    np.random.seed(seed)

    # ====================================================
    # FIXED HETEROGENEITY FIELDS
    # ====================================================
    gamma_base = np.random.uniform(0.05, 0.34, (Nx, Ny)) * (1 + 2*np.random.randn(Nx, Ny))
    I0_base = np.random.uniform(0.01, 0.15, (Nx, Ny))
    tau_base = np.random.uniform(0.5, 1.1, (Nx, Ny))
    D0_base = np.random.uniform(0.05, 0.5, (Nx, Ny))
    kappa_base = np.random.uniform(1, 4, (Nx, Ny))

    Sc_values = []
    chi_values = []
    corr_lengths = []
    snapshots = {}

    for idx, alpha in enumerate(alpha_values):

        print(f"{'Disease' if disease else 'Healthy'} | alpha = {alpha:.3f}")

        # ----------------------------------------------------
        # SAME HETEROGENEITY FOR ALL ATP LEVELS
        # ----------------------------------------------------
        gamma = gamma_base.copy()
        I0 = 0.05 + (1 / np.sqrt(alpha)) * I0_base
        tau_h = 10.0 / ((1 + 0.8 * alpha) * tau_base)
        D0 = D0_base.copy()
        kappa = kappa_base.copy()

        # ----------------------------------------------------
        # DISEASE MODIFICATIONS
        # ----------------------------------------------------
        if disease:
            gamma *= 2.0
            tau_h *= 3.0
            D0 *= 0.5
            kappa *= 1.5

        # ----------------------------------------------------
        # EFFECTIVE DIFFUSION
        # ----------------------------------------------------
        Deff = D0 / (1 + (kappa * alpha)**4)

        # ATP-dependent threshold
        theta = theta_base + 0.7 * alpha

        # ----------------------------------------------------
        # INITIAL CONDITIONS
        # ----------------------------------------------------
        if idx == 0:
            C = np.random.uniform(-0.1, 0.3, (Nx, Ny))
            h = np.random.uniform(0.4, 1.2, (Nx, Ny))

        C_hist = []

        # ====================================================
        # TIME LOOP
        # ====================================================
        for t in range(steps):

            sigma_eff = sigma * (1 + 4 * alpha)
            noise = sigma_eff * 3 * np.random.randn(Nx, Ny)

            C_active = Phi(C, theta)
            diff = Deff * laplacian(C_active)

            dC = (C - (C**3)/3 - h + I0 + gamma * alpha + diff)
            dh = (C + a - b*h) / tau_h

            C += dt * dC + dt ** 0.5 * noise
            h += dt * dh

            C = np.clip(C, -4, 4)
            C_hist.append(C.copy())

        # ====================================================
        # POSTPROCESSING
        # ====================================================
        C_hist = np.array(C_hist)
        C_stat = C_hist[int(0.3 * len(C_hist)):]
        flat = C_stat.reshape(len(C_stat), -1)
        DeltaC = (np.max(flat, axis=1) - np.min(flat, axis=1))

        # ----------------------------------------------------
        # Spatial heterogeneity
        # ----------------------------------------------------
        spatial_var = np.var(C_hist, axis=(1,2))
        spatial_mean = np.mean(C_hist, axis=(1,2))
        Sc_t = spatial_var / np.abs(spatial_mean)

        # ----------------------------------------------------
        # Extreme fluctutuation
        # ----------------------------------------------------
        chi = N * (np.mean(DeltaC**2) - np.mean(DeltaC)**2)

        # ----------------------------------------------------
        # Correlation length proxy
        # ----------------------------------------------------
        last = np.mean(C_stat[-5:], axis=0).flatten()
        corr = np.correlate(last - last.mean(), last - last.mean(), mode='full')
        corr = corr[corr.size // 2:]
        corr = corr / corr[0]
        corr_length = np.sum(corr > 0.2)

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------
        Sc_values.append(np.mean(Sc_t))
        chi_values.append(chi)
        corr_lengths.append(corr_length)

        # ----------------------------------------------------
        # SNAPSHOTS
        # ----------------------------------------------------
        if np.isclose(alpha, target_snaps[0]): snapshots["low"] = C_stat[-1]
        if np.isclose(alpha, target_snaps[1]): snapshots["mid"] = C_stat[-1]
        if np.isclose(alpha, target_snaps[2]): snapshots["high"] = C_stat[-1]

    return {
        "Sc": np.array(Sc_values),
        "chi": np.array(chi_values),
        "corr": np.array(corr_lengths),
        "snapshots": snapshots
    }
# ============================================================
# RUN BOTH MODELS
# ============================================================
healthy = run_model(disease=False, seed=11)
disease = run_model(disease=True, seed=11)

# 49 [0.175, 0.285, 1.0]
# 29 [0.175, 0.285, 1.0]  
# 146 [0.175, 0.285, 1.0]  
# 145 [0.23, 0.45, 1.0]
# 65 74 11 kötü
# 76 [0.175, 0.285, 1.0]
# 22 [0.175, 0.285, 1.0]
# 55 [0.175, 0.505, 1.0]
# 57 [0.23, 0.56, 1.0]
# ============================================================
# SMOOTHING
# ============================================================
def smooth(x, w):
    return np.convolve(x, np.ones(w)/w, mode='same')

healthy_Sc   = smooth(healthy["Sc"], 3)
healthy_chi  = smooth(healthy["chi"], 3)
healthy_corr = smooth(healthy["corr"], 5)

disease_Sc   = smooth(disease["Sc"], 3)
disease_chi  = smooth(disease["chi"], 3)
disease_corr = smooth(disease["corr"], 5)

# ============================================================
# FIGURE
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# ============================================================
# FIRST ROW : CURVES
# ============================================================
# ------------------------------------------------------------
# Spatial heterogeneity
# ------------------------------------------------------------
axes[0,0].plot(alpha_values,healthy_Sc,linewidth=2,label='Healthy')
axes[0,0].plot(alpha_values,disease_Sc,linewidth=2,label='Disease')
axes[0,0].set_title("(A) Spatial Heterogeneity $S_c$")
axes[0,0].set_xlabel(r'ATP level $\alpha$')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend()

# ------------------------------------------------------------
# Susceptibility
# ------------------------------------------------------------
axes[0,1].plot(alpha_values,healthy_chi,linewidth=2,label='Healthy')
axes[0,1].plot(alpha_values,disease_chi,linewidth=2,label='Disease')
axes[0,1].set_title(r"(B) Extreme-Value Fluctuation $\chi_{ext}$")
axes[0,1].set_xlabel(r'ATP level $\alpha$')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend()

# ------------------------------------------------------------
# Correlation length
# ------------------------------------------------------------
axes[0,2].plot(alpha_values,healthy_corr,linewidth=2,label='Healthy')
axes[0,2].plot(alpha_values,disease_corr,linewidth=2,label='Disease')
axes[0,2].set_title(r"(C) Spatial Coherence Length Proxy $\xi$")
axes[0,2].set_xlabel(r'ATP level $\alpha$')
axes[0,2].grid(True, alpha=0.3)
axes[0,2].legend()

# ============================================================
# SECOND ROW : HEALTHY SNAPSHOTS
# ============================================================

healthy_titles = ["(D1) Healthy Low ATP", "(D2) Healthy Mid ATP", "(D3) Healthy High ATP"]

healthy_keys = ["low", "mid", "high"]

for i, key in enumerate(healthy_keys):

    # nearest, none, antialiased, kaiser

    #im = axes[1,i].imshow(healthy["snapshots"][key], cmap='inferno', origin='lower')
    im = axes[1,i].imshow(healthy["snapshots"][key], cmap='inferno', origin='lower', interpolation='nearest')
    axes[1,i].set_xlabel(healthy_titles[i])
    axes[1,i].set_xticks([])
    axes[1,i].set_yticks([])

# ============================================================
# THIRD ROW : DISEASE SNAPSHOTS
# ============================================================

disease_titles = ["(E1) Disease Low ATP", "(E2) Disease Mid ATP", "(E3) Disease High ATP"]

disease_keys = ["low", "mid", "high"]

for i, key in enumerate(disease_keys):

    im2 = axes[2,i].imshow(disease["snapshots"][key], cmap='inferno', origin='lower', interpolation='nearest')
    #im2 = axes[2,i].imshow(disease["snapshots"][key], cmap='inferno', origin='lower')

    axes[2,i].set_xlabel(disease_titles[i])
    axes[2,i].set_xticks([])
    axes[2,i].set_yticks([])

# ============================================================
# COLORBAR
# ============================================================

cax = fig.add_axes([0.05, 0.15, 0.015, 0.4])

cbar = fig.colorbar(im2, cax=cax)
cbar.set_label("Calcium activity (C)")

#plt.tight_layout()
plt.show()

