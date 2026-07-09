import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from plotstyle import apply_style, save_fig, panel_label

# ============================================================
# OPERATORS
# ============================================================
@njit(fastmath=True)
def laplacian(Z):
    Nx, Ny = Z.shape
    L = np.empty_like(Z)

    for i in range(Nx):
        ip = (i + 1) % Nx
        im = (i - 1) % Nx

        for j in range(Ny):
            jp = (j + 1) % Ny
            jm = (j - 1) % Ny

            L[i,j] = (Z[ip,j] + Z[im,j] + Z[i,jp] + Z[i,jm] - 4.0*Z[i,j])

    return L


@njit(fastmath=True)
def Phi(C, theta, eta=8.0):
    return 0.5*(1.0 + np.tanh(eta*(C-theta)))

# ============================================================
# DISORDER
# ============================================================
def generate_disorder(Nx, Ny, seed=0):

    rng = np.random.default_rng(seed)

    return {
        "gamma": rng.uniform(0.05, 0.34, (Nx, Ny)),
        "I0_base": rng.uniform(0.01, 0.15, (Nx, Ny)),
        "tau_base": rng.uniform(0.5, 1.1, (Nx, Ny)),
        "D0": rng.uniform(0.05, 0.5, (Nx, Ny)),
        "kappa": rng.uniform(1.0, 4.0, (Nx, Ny)),
    }

# ============================================================
# ROBUSTNESS
# ============================================================
def robustness_SC(sc_runs):

    sc_runs = np.asarray(sc_runs)
    mu = np.mean(sc_runs)
    sd = np.std(sc_runs)

    return mu/sd if sd > 1e-12 else np.nan

# ============================================================
# SPATIAL HETEROGENEITY
# ============================================================
def spatial_heterogeneity(C):
    
    return np.var(C) / (np.abs(np.mean(C)) + 1e-9)

# ============================================================
# LARGEST LYAPUNOV EXPONENT
# ============================================================
@njit(fastmath=True)
def largest_lyapunov_core(noise_all,C1, h1,
    C2, h2,I0,gamma,alpha,Deff,tau_h,theta,dt,tau,T):

    eps = 1e-8
    log_sum = 0.0

    steps = noise_all.shape[0]

    for t in range(steps):

        noise = noise_all[t]

        dC1 = (C1 - C1**3/3.0 - h1 + I0 + gamma*alpha + Deff*laplacian(Phi(C1,theta)) + noise)
        dh1 = (C1 + 1.0 - 0.8*h1)/tau_h

        dC2 = (C2 - C2**3/3.0 - h2 + I0 + gamma*alpha + Deff*laplacian(Phi(C2,theta)) + noise)
        dh2 = (C2 + 1.0 - 0.8*h2)/tau_h

        C1 += dt*dC1
        h1 += dt*dh1

        C2 += dt*dC2
        h2 += dt*dh2

        if t > 0 and t % tau == 0:

            delta = np.sqrt(np.mean((C1-C2)**2) + np.mean((h1-h2)**2))

            #if delta < 1e-14:
               # delta = 1e-14

            log_sum += np.log(delta/eps)

            C2 = C1 + (C2-C1)/delta * eps
            h2 = h1 + (h2-h1)/delta * eps

    M = steps // tau
    return log_sum / (M * tau * dt)

# ============================================================
# 1) TRUE BENETTIN LLE
# ============================================================
def largest_lyapunov(Nx, Ny, alpha, disorder, disease=False, 
                 T=200, 
                 dt=0.0067, 
                 tau=10):

    rng = np.random.default_rng(123)
    steps = int(T/dt) 

    gamma = disorder["gamma"].copy()
    I0_base = disorder["I0_base"]
    tau_base = disorder["tau_base"]
    D0 = disorder["D0"].copy()
    kappa = disorder["kappa"].copy()

    I0 = 0.05 + (1/np.sqrt(alpha))*I0_base
    tau_h = 10/((1+0.8*alpha)*tau_base)

    if disease:
        gamma *= 2
        tau_h *= 3
        D0 *= 0.5
        kappa *= 1.5

    Deff = D0/(1+(kappa*alpha)**4)
    theta = 0.5 + 0.7*alpha

    C1 = rng.uniform(-0.1,0.3,(Nx,Ny))
    h1 = rng.uniform(0.4,1.2,(Nx,Ny))
    C2 = C1.copy()
    h2 = h1.copy()
    C2 += 1e-8 * rng.standard_normal((Nx,Ny))
    h2 += 1e-8 * rng.standard_normal((Nx,Ny))

    log_sum = 0.0

    noise_all = rng.standard_normal((steps, Nx, Ny))

    return largest_lyapunov_core(noise_all,C1, h1,C2, h2,I0,
    gamma,alpha,Deff, tau_h,theta,dt,tau,T)

# ============================================================
# 2) BOOTSTRAP ERROR BAR
# ============================================================
def bootstrap(values):

    data = np.array(values)

    return np.mean(data), np.std(data)

# ============================================================
# SINGLE SC RUN (NUMBA)
# ============================================================
@njit(fastmath=True)
def single_SC_core(Nx,Ny,alpha,gamma,I0_base,tau_base,D0,kappa,disease,seed,T,dt):

    np.random.seed(seed)

    steps = int(T / dt)

    sigma = 0.4
    a = 1.0
    b = 0.8

    gamma_local = gamma.copy()
    D0_local = D0.copy()
    kappa_local = kappa.copy()

    I0 = 0.05 + (1.0 / np.sqrt(alpha)) * I0_base
    tau_h = 10.0 / ((1.0 + 0.8 * alpha) * tau_base)

    if disease:
        gamma_local *= 2.0
        tau_h *= 3.0
        D0_local *= 0.5
        kappa_local *= 1.5

    Deff = D0_local / (1.0 + (kappa_local * alpha) ** 4)

    theta = 0.5 + 0.7 * alpha

    C = np.random.uniform(-0.1, 0.3, (Nx, Ny))
    h = np.random.uniform(0.4, 1.2, (Nx, Ny))

    noise_amp = sigma * (1.0 + 4.0 * alpha)

    for _ in range(steps):

        noise = noise_amp * np.random.standard_normal((Nx, Ny))

        dC = (C - C**3 / 3.0 - h + I0 + gamma_local * alpha
            + Deff * laplacian(Phi(C, theta)) + noise)

        dh = (C + a - b * h) / tau_h

        C += dt * dC
        h += dt * dh

    return np.var(C) / (np.abs(np.mean(C)) + 1e-12)


def single_SC(Nx, Ny,alpha,disorder,disease=False,
    seed=0,
    T=200,
    dt=0.007
):

    return single_SC_core(Nx,Ny,alpha,disorder["gamma"],disorder["I0_base"],
        disorder["tau_base"],disorder["D0"],disorder["kappa"],disease,seed,T,dt)

# ============================================================
# PARAMETERS
# ============================================================
sizes = [(5, 5), (10, 10), (15, 15), (20, 20)]

alphas = np.linspace(0.01, 1.1, 21)

Nruns = 23

analysis = {}

# ============================================================
# MAIN ANALYSIS
# ============================================================

for Nx, Ny in sizes:

    print(f"\nSize {Nx}x{Ny}")

    disorder = generate_disorder(Nx,Ny,seed=1000 + Nx)

    lambda_h = []
    lambda_d = []

    RSC_h = []
    RSC_d = []
    
    for alpha in alphas:

        print(f"alpha = {alpha:.3f}")

        lambda_h.append(largest_lyapunov(Nx,Ny,alpha,disorder,disease=False))
        lambda_d.append(largest_lyapunov(Nx,Ny,alpha,disorder,disease=True))

        sc_runs_h = []
        sc_runs_d = []

        for k in range(Nruns):

            seed = 100 + k

            sc_runs_h.append(single_SC(Nx,Ny,alpha,disorder,disease=False,seed=seed))
            sc_runs_d.append(single_SC(Nx,Ny,alpha,disorder,disease=True,seed=seed))

        RSC_h.append(robustness_SC(sc_runs_h))
        RSC_d.append(robustness_SC(sc_runs_d))

    analysis[(Nx, Ny)] = {

        "lambda_h": np.array(lambda_h),
        "lambda_d": np.array(lambda_d),

        "RSC_h": np.array(RSC_h),
        "RSC_d": np.array(RSC_d),
    }

# ============================================================
# PLOTS
# ============================================================
apply_style()
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

# ------------------------------------------------------------
# LLE
# ------------------------------------------------------------
for size, data in analysis.items():

    line, = ax[0].plot(alphas, data["lambda_h"], '-', label=f'{size}')
    ax[0].plot(alphas, data["lambda_d"], '--',
           color=line.get_color(),
           label='_nolegend_')
    ax[0].axhline(0,color='k',linestyle=':')
    ax[0].set_title("Largest Lyapunov Exponent")
    panel_label(ax[0], "A")
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("Lambda")
    ax[0].legend(fontsize=8)

# ------------------------------------------------------------
# ROBUSTNESS
# ------------------------------------------------------------
for size, data in analysis.items():

    line, = ax[2].plot(alphas, data["RSC_h"], '-', label=f'H {size}')
    ax[2].plot(alphas, data["RSC_d"], '--',
           color=line.get_color(),
           label='_nolegend_')

    ax[2].set_title(r"$R_{SC}=\langle S_C\rangle/\sigma(S_C)$")
    panel_label(ax[2], "C")
    ax[2].set_xlabel("Alpha")
    ax[2].set_ylabel("RSC")
    ax[2].legend(fontsize=8)

# ------------------------------------------------------------
# DELTA LAMBDA
# ------------------------------------------------------------
for size, data in analysis.items():

    delta_lambda = (data["lambda_d"] - data["lambda_h"])
    ax[1].plot(alphas,delta_lambda,label=str(size))
    ax[1].axhline(0,color='k',linestyle=':')
    ax[1].set_title(r"$\Delta \lambda = \lambda_D-\lambda_H$")
    panel_label(ax[1], "B")
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Delta Lambda")
    ax[1].legend(fontsize=8)

fig.tight_layout()
save_fig(fig, "Figure_4")
print("Saved Figure_4.pdf / .png")
plt.show()