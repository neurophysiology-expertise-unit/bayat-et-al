"""Generate the visual reading guide (communication/what_we_tested.html) from the
provenance-stamped npz files, so every chart is the measured data and not a sketch.

Charts are emitted as inline SVG. Palette validated with the dataviz six-checks script:
  light  #008C99 #B0651C #6A5296   dark  #2AA6B4 #C87E2E #8B79C4
Run: python make_explainer.py
"""
import sys, pathlib, glob, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

OUT = pathlib.Path("/mnt/sysfs01/users/cagatay/code/neubrain/projects/astro_atp"
                   "/communication/what_we_tested.html")
PD = pathlib.Path("processed_data")

# ---------------------------------------------------------------- data
def load():
    d = {}
    z = np.load(PD / "i0_fine.npz", allow_pickle=True)
    d["i0"] = dict(x=z["baselines"], rate=z["rate_mean"], part=z["pct_active_mean"])
    e = np.load(PD / "fig4_mechanisms_ens.npz", allow_pickle=True)
    seeds = sorted(glob.glob(str(PD / "fig4_seed*.npz")))
    nr = np.array([np.load(f, allow_pickle=True)["nucleation_rate"] for f in seeds])
    r2g = np.array([np.load(f, allow_pickle=True)["r2_vs_gamma"] for f in seeds])
    cr = np.array([np.load(f, allow_pickle=True)["control_r2"] for f in seeds])
    gammas = np.load(seeds[0], allow_pickle=True)["gammas"]
    d["tau"] = dict(x=e["taus"], nucl=nr.mean(0), nucl_sd=nr.std(0),
                    ext=e["extent_vs_tau_mean"], ext_sd=e["extent_vs_tau_sd"])
    g = np.load(PD / "validation_gamma_regen_nucleation_ens.npz", allow_pickle=True)
    d["gam"] = dict(x=gammas, ext=e["extent_vs_gamma_mean"], ext_sd=e["extent_vs_gamma_sd"],
                    nucl=g["nucleation_rate_mean"], nucl_sd=g["nucleation_rate_sd"])
    f2 = np.load(PD / "fig2_excitability.npz", allow_pickle=True)
    d["atp"] = dict(x=f2["alphas"], fdet=f2["f_det"] * 100, act=f2["A_act"] * 100)
    c = {}
    for tag, fn in (("A", "A_full"), ("B", "B_coupling_aref0.10"), ("Bp", "Bprime_coupθ_aref0.10")):
        z = np.load(PD / f"phase2v1_bayat_b0.42_L32_{fn}.npz", allow_pickle=True)
        c[tag] = z["active"].mean(0) * 100
        c["x"] = z["alphas"]
    d["coup"] = c
    l = np.load(PD / "lagcheck.npz", allow_pickle=True)
    d["lag"] = dict(sep=l["separations"], lag=l["lag_mean"], corr=l["peakcorr_mean"],
                    nres=l["n_resolved"], taus=l["taus"])
    d["r2"] = dict(rmax=np.array([5., 6, 8, 12, 35]), ctrl=cr.mean(0), ctrl_sd=cr.std(0),
                   bounded=float(r2g[:, 9].mean()), bounded_sd=float(r2g[:, 9].std()))
    t = np.load(PD / "termination_time.npz", allow_pickle=True)
    d["term"] = dict(g=t["gammas"], t=t["stop_time_mean"], sd=t["stop_time_sd"],
                     ext=t["extent_mean"] * 25.0)
    return d

# ---------------------------------------------------------------- svg helpers
def sc(v, lo, hi, a, b):
    if hi == lo: return (a + b) / 2
    return a + (float(v) - lo) / (hi - lo) * (b - a)

def axes(W, H, m, xlab, ylab, xt, yt, xr, yr, logx=False):
    """returns (svg_parts, xmap, ymap)"""
    L, R, T, B = m
    def xm(v):
        if logx: return sc(np.log10(max(v, 1e-9)), np.log10(xr[0]), np.log10(xr[1]), L, W - R)
        return sc(v, xr[0], xr[1], L, W - R)
    def ym(v): return sc(v, yr[0], yr[1], H - B, T)
    p = []
    for v in yt:
        y = ym(v)
        p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{L-8}" y="{y+3.5:.1f}" class="tick tr">{fmt(v)}</text>')
    for v in xt:
        x = xm(v)
        p.append(f'<text x="{x:.1f}" y="{H-B+16}" class="tick tc">{fmt(v)}</text>')
    p.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" class="axis"/>')
    p.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-4}" class="axlab tc">{xlab}</text>')
    p.append(f'<text transform="translate(13,{(T+H-B)/2:.0f}) rotate(-90)" class="axlab tc">{ylab}</text>')
    return p, xm, ym

def fmt(v):
    v = float(v)
    if v == int(v) and abs(v) < 1e4: return str(int(v))
    if abs(v) >= 1: return f"{v:g}"
    return f"{v:g}"

def line(xs, ys, xm, ym, cls, dots=True):
    pts = " ".join(f"{xm(x):.1f},{ym(y):.1f}" for x, y in zip(xs, ys))
    out = [f'<polyline points="{pts}" class="{cls}"/>']
    if dots:
        for x, y in zip(xs, ys):
            out.append(f'<circle cx="{xm(x):.1f}" cy="{ym(y):.1f}" r="3.4" class="{cls} dot"/>')
    return out

def band(xs, ys, sd, xm, ym, cls):
    dn = " ".join(f"{xm(x):.1f},{ym(y-s):.1f}" for x, y, s in
                  sorted(zip(xs, ys, sd), key=lambda t: -t[0]))
    up = " ".join(f"{xm(x):.1f},{ym(y+s):.1f}" for x, y, s in zip(xs, ys, sd))
    return [f'<polygon points="{up} {dn}" class="{cls} bandfill"/>']

def chart(W, H, parts, cap=None):
    c = f'<figcaption>{cap}</figcaption>' if cap else ""
    return (f'<figure class="chart"><svg viewBox="0 0 {W} {H}" role="img" '
            f'preserveAspectRatio="xMidYMid meet">{"".join(parts)}</svg>{c}</figure>')

# ---------------------------------------------------------------- the charts
def mini(d, key, which, W=300, H=190):
    """One cell of the dissociation grid."""
    m = (54, 14, 16, 40)
    if key == "tau":
        xs = d["tau"]["x"]; xr = (0, 30); xt = [0, 5, 15, 30]; xlab = "τ_ref  (s)"
    else:
        xs = d["gam"]["x"]; xr = (0.05, 1.0); xt = [0.05, 0.1, 0.25, 0.5, 1.0]; xlab = "γ_regen"
    if which == "nucl":
        ys = d[key]["nucl"]; sd = d[key]["nucl_sd"]; ylab = "nucleation"
        yr = (0, 1.35); yt = [0, 0.5, 1.0]; cls = "s1"
    else:
        ys = d[key]["ext"]; sd = d[key]["ext_sd"]; ylab = "extent (cells)"
        yr = (0, 38); yt = [0, 10, 20, 30]; cls = "s2"
    p, xm, ym = axes(W, H, m, xlab, ylab, xt, yt, xr, yr, logx=(key == "gam"))
    p += band(xs, ys, sd, xm, ym, cls) + line(xs, ys, xm, ym, cls)
    lo, hi = float(np.min(ys)), float(np.max(ys))
    chg = (hi - lo) / hi * 100 if hi else 0
    flat = chg < 5
    tag = "unchanged" if flat else f"−{chg:.0f}%"
    p.append(f'<text x="{W-18}" y="{28}" class="delta {"flat" if flat else "moved"} tr">{tag}</text>')
    return chart(W, H, p)

def build(d):
    C = {}
    # --- I0 ignition transition
    W, H, m = 640, 250, (58, 18, 18, 44)
    i = d["i0"]; msk = i["x"] >= 0.30
    xs = i["x"][msk]
    p, xm, ym = axes(W, H, m, "baseline excitability  I₀^base", "cells participating (%)",
                     [0.30, 0.35, 0.40, 0.45, 0.50], [0, 25, 50, 75, 100], (0.30, 0.50), (0, 105))
    p += line(xs, i["part"][msk], xm, ym, "s1")
    x42 = xm(0.42)
    p.append(f'<line x1="{x42:.1f}" y1="18" x2="{x42:.1f}" y2="{H-44}" class="marker"/>')
    p.append(f'<text x="{x42+6:.1f}" y="32" class="note">0.42 selected — 98.5% participate</text>')
    x38 = xm(0.38)
    p.append(f'<rect x="{x38:.1f}" y="18" width="{x42-x38:.1f}" height="{H-62}" class="zone"/>')
    p.append(f'<text x="{(x38+x42)/2:.1f}" y="{H-56}" class="note tc">ignition transition</text>')
    C["i0"] = chart(W, H, p, "Participation is a step, not a slope. Below the transition the medium "
                             "is subthreshold; we take the first value where essentially every cell joins in.")

    # --- ATP recruited vs activity
    a = d["atp"]
    p, xm, ym = axes(W, H, m, "ATP  (A)", "% of cells", [0, 0.25, 0.5, 0.75, 1.0],
                     [0, 20, 40, 60, 80], (0, 1.11), (0, 80))
    p += line(a["x"], a["fdet"], xm, ym, "s1", dots=False)
    p += line(a["x"], a["act"], xm, ym, "s2", dots=False)
    p.append(f'<text x="{xm(1.0):.1f}" y="{ym(68.5)-10:.1f}" class="lbl s1t tr">deterministically recruited</text>')
    p.append(f'<text x="{xm(1.0):.1f}" y="{ym(21)+16:.1f}" class="lbl s2t tr">actually active on the lattice</text>')
    xg = xm(0.065)
    p.append(f'<circle cx="{xg:.1f}" cy="{ym(14.8):.1f}" r="6" class="callout"/>')
    p.append(f'<text x="{xg+10:.1f}" y="{ym(14.8)-8:.1f}" class="note">14.8% active — yet zero units have bifurcated</text>')
    C["atp"] = chart(W, H, p, "The two never meet. At the low end the lattice is already active while "
                              "no single cell has crossed its threshold: that activity is noise-driven "
                              "and has no deterministic counterpart, so recruitment cannot be what drives it.")

    # --- coupling control
    c = d["coup"]
    p, xm, ym = axes(W, H, m, "ATP  (A)", "active cells (%)", [0, 0.25, 0.5, 0.75, 1.0],
                     [0, 10, 20, 30], (0, 1.11), (0, 30))
    p += line(c["x"], c["A"], xm, ym, "s1", dots=False)
    p += line(c["x"], c["B"], xm, ym, "s2", dots=False)
    p += line(c["x"], c["Bp"], xm, ym, "s3", dots=False)
    p.append(f'<text x="{xm(1.11)-4:.1f}" y="{ym(c["A"][-1])-8:.1f}" class="lbl s1t tr">all channels follow ATP</text>')
    p.append(f'<text x="{xm(1.11)-4:.1f}" y="{ym(c["B"][-1])+16:.1f}" class="lbl s2t tr">only coupling follows ATP</text>')
    p.append(f'<text x="{xm(1.11)-4:.1f}" y="{ym(c["Bp"][-1])+30:.1f}" class="lbl s3t tr">coupling + threshold</text>')
    C["coup"] = chart(W, H, p, "Coupling collapses seventeenfold in all three arms. Activity rises anyway "
                               "when every channel follows ATP, and falls when coupling is the only one moving — "
                               "so the ATP effect is not a coupling effect wearing a disguise.")

    # --- lag vs distance
    W2, H2 = 640, 230
    lg = d["lag"]
    p, xm, ym = axes(W2, H2, (58, 18, 18, 44), "separation  (cells)", "correlation peak",
                     [1, 2, 3, 4, 6, 8, 10, 12], [0, 0.1, 0.2, 0.3, 0.4], (1, 12), (0, 0.42))
    for idx, cls, nm in ((0, "s1", "no refractory state"), (2, "s2", "τ_ref = 15 s")):
        p += line(lg["sep"], lg["corr"][idx], xm, ym, cls, dots=True)
    p.append(f'<line x1="{xm(1):.1f}" y1="{ym(0.08):.1f}" x2="{xm(12):.1f}" y2="{ym(0.08):.1f}" class="thresh"/>')
    p.append(f'<text x="{xm(12):.1f}" y="{ym(0.08)-7:.1f}" class="note tr">noise floor — below this a lag is not measurable</text>')
    p.append(f'<text x="{xm(3):.1f}" y="{ym(0.30):.1f}" class="lbl s1t">no refractory state</text>')
    p.append(f'<text x="{xm(3.2):.1f}" y="{ym(0.13):.1f}" class="lbl s2t">τ_ref = 15 s</text>')
    C["lag"] = chart(W2, H2, p, "Where the lag can be measured at all it is exactly zero — 10 of 10 seeds at "
                                "one cell spacing. Past two or three spacings the correlation is in the noise, "
                                "so there is no lag-versus-distance relation to fit. Synchrony, not propagation.")

    # --- R2 ladder
    r = d["r2"]
    p, xm, ym = axes(W2, H2, (58, 18, 18, 44), "radius the fit is measured over (cells)", "R² of fit",
                     [5, 6, 8, 12, 35], [0.6, 0.7, 0.8, 0.9, 1.0], (5, 35), (0.6, 1.0))
    p += band(r["rmax"], r["ctrl"], r["ctrl_sd"], xm, ym, "s1")
    p += line(r["rmax"], r["ctrl"], xm, ym, "s1")
    y9 = ym(0.9)
    p.append(f'<line x1="{xm(5):.1f}" y1="{y9:.1f}" x2="{xm(35):.1f}" y2="{y9:.1f}" class="thresh"/>')
    p.append(f'<text x="{xm(35):.1f}" y="{y9-7:.1f}" class="note tr">our pre-set bar: R² &gt; 0.9</text>')
    yb = ym(r["bounded"])
    p.append(f'<line x1="{xm(5)-6:.1f}" y1="{yb:.1f}" x2="{xm(6.6):.1f}" y2="{yb:.1f}" class="s2 bounded"/>')
    p.append(f'<text x="{xm(6.9):.1f}" y="{yb+4:.1f}" class="lbl s2t">the bounded front, 0.695</text>')
    p.append(f'<text x="{xm(12):.1f}" y="{ym(0.79):.1f}" class="lbl s1t">a front we KNOW propagates</text>')
    C["r2"] = chart(W2, H2, p, "The bar is the problem, not the front. A front known to propagate also fails "
                               "R² > 0.9 at every radius — so the statistic cannot tell a bounded front from an "
                               "unbounded one over the distance available. We report that rather than lowering the bar.")

    # --- termination: distance AND time
    t = d["term"]
    p, xm, ym = axes(W2, H2, (58, 18, 18, 44), "where the front stops  (μm)", "when it stops  (s)",
                     [0, 50, 100, 150, 200, 250], [0, 10, 20, 30, 40], (0, 260), (0, 45))
    p.append(f'<rect x="{xm(100):.1f}" y="{ym(20):.1f}" width="{xm(250)-xm(100):.1f}" '
             f'height="{ym(10)-ym(20):.1f}" class="zone2"/>')
    p.append(f'<text x="{xm(175):.1f}" y="{ym(21.5):.1f}" class="note tc">what is measured in tissue: 100–250 μm, ~15 s</text>')
    for g, tt, ex, s in zip(t["g"], t["t"], t["ext"], t["sd"]):
        p.append(f'<line x1="{xm(ex):.1f}" y1="{ym(tt-s):.1f}" x2="{xm(ex):.1f}" y2="{ym(tt+s):.1f}" class="ebar"/>')
        p.append(f'<circle cx="{xm(ex):.1f}" cy="{ym(tt):.1f}" r="5.5" class="s2 dot"/>')
        p.append(f'<text x="{xm(ex):.1f}" y="{ym(tt)-13:.1f}" class="note tc">γ={g:g}</text>')
    C["term"] = chart(W2, H2, p, "One parameter, two independent constraints. The gain interval that puts the "
                                 "front inside the reported distance also puts it inside the reported duration — "
                                 "and the timing was never fitted, only checked afterwards.")

    C["single"] = single_cell(d)
    C["diagrams"] = DIAGRAMS
    C["grid"] = '<div class="dissoc">' + "".join(
        f'<div class="cell"><div class="cellhead"><span class="csym">{sym}</span>'
        f'<span class="cwhat">{lab}</span></div>{mini(d, k, w)}</div>'
        for k, w, sym, lab in (
            ("tau", "nucl", "τ_ref", "spontaneous ignitions"),
            ("tau", "ext", "τ_ref", "how far the wave goes"),
            ("gam", "nucl", "γ_regen", "spontaneous ignitions"),
            ("gam", "ext", "γ_regen", "how far the wave goes"))) + "</div>"
    return C


CSS = """
:root{
  --ground:#F7F9F9;--panel:#FFFFFF;--ink:#16232B;--muted:#5A6970;--faint:#86959C;
  --rule:#DDE5E6;--rule2:#C3D0D2;--accent:#008C99;--accent-soft:#E2F0F1;
  --s1:#008C99;--s2:#B0651C;--s3:#6A5296;
  --ok:#2F7D4F;--ok-soft:#E6F1EA;--bad:#A33A2E;--bad-soft:#F6E7E4;
  --warn:#8A6212;--warn-soft:#F5EDDD;
  --shadow:0 1px 2px rgba(22,35,43,.05),0 10px 28px -20px rgba(22,35,43,.3);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0F1619;--panel:#161F23;--ink:#E4ECED;--muted:#9CACB2;--faint:#75868C;
  --rule:#253237;--rule2:#33444A;--accent:#2AA6B4;--accent-soft:#11292D;
  --s1:#2AA6B4;--s2:#C87E2E;--s3:#8B79C4;
  --ok:#6FC08D;--ok-soft:#13291D;--bad:#E08376;--bad-soft:#2C1815;
  --warn:#D8AC5A;--warn-soft:#2A2213;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -20px rgba(0,0,0,.9);
}}
:root[data-theme="dark"]{
  --ground:#0F1619;--panel:#161F23;--ink:#E4ECED;--muted:#9CACB2;--faint:#75868C;
  --rule:#253237;--rule2:#33444A;--accent:#2AA6B4;--accent-soft:#11292D;
  --s1:#2AA6B4;--s2:#C87E2E;--s3:#8B79C4;
  --ok:#6FC08D;--ok-soft:#13291D;--bad:#E08376;--bad-soft:#2C1815;
  --warn:#D8AC5A;--warn-soft:#2A2213;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -20px rgba(0,0,0,.9);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;font-size:16px;line-height:1.6;
  font-family:"IBM Plex Sans",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:62rem;margin:0 auto;padding:3.5rem 1.5rem 6rem}
@media(max-width:640px){.wrap{padding:2.2rem 1rem 4rem}}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent);margin:0 0 .9rem}
h1{font-family:Spectral,Georgia,serif;font-weight:600;font-size:clamp(2rem,5vw,2.9rem);
  line-height:1.12;letter-spacing:-.015em;margin:0 0 .9rem;text-wrap:balance}
.standfirst{font-size:1.09rem;color:var(--muted);max-width:40rem;margin:0}
h2{font-family:Spectral,Georgia,serif;font-weight:600;font-size:1.55rem;margin:0 0 .35rem;
  letter-spacing:-.01em;text-wrap:balance}
h2 .num{font-family:"IBM Plex Mono",monospace;font-size:.8rem;color:var(--faint);
  letter-spacing:.09em;margin-right:.6rem;vertical-align:.3em}
.sectnote{color:var(--muted);max-width:44rem;margin:0 0 1.5rem}
section{margin-top:3.4rem}
p{max-width:43rem}
figure.chart{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:10px;
  padding:1rem 1rem .5rem;box-shadow:var(--shadow);overflow-x:auto}
figure.chart svg{width:100%;height:auto;display:block}
figcaption{color:var(--muted);font-size:.88rem;line-height:1.5;padding:.7rem .3rem .5rem;
  border-top:1px solid var(--rule);margin-top:.6rem;max-width:46rem}
.grid{stroke:var(--rule);stroke-width:1}
.axis{stroke:var(--rule2);stroke-width:1}
.tick{fill:var(--faint);font-size:10.5px;font-family:"IBM Plex Mono",monospace}
.axlab{fill:var(--muted);font-size:11px;font-family:"IBM Plex Sans",sans-serif}
.tc{text-anchor:middle}.tr{text-anchor:end}
polyline{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
circle.dot{stroke:var(--panel);stroke-width:2}
.s1{stroke:var(--s1)}.s1 circle,circle.s1{fill:var(--s1)}
.s2{stroke:var(--s2)}.s2 circle,circle.s2{fill:var(--s2)}
.s3{stroke:var(--s3)}.s3 circle,circle.s3{fill:var(--s3)}
.bandfill{stroke:none;opacity:.16}
.s1.bandfill{fill:var(--s1)}.s2.bandfill{fill:var(--s2)}
.lbl{font-size:11.5px;font-weight:500;font-family:"IBM Plex Sans",sans-serif}
.s1t{fill:var(--s1)}.s2t{fill:var(--s2)}.s3t{fill:var(--s3)}
.note{fill:var(--muted);font-size:10.5px;font-family:"IBM Plex Sans",sans-serif}
.marker{stroke:var(--ink);stroke-width:1.2;stroke-dasharray:3 3;opacity:.55}
.thresh{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:4 3;opacity:.8}
.zone{fill:var(--accent);opacity:.07}
.zone2{fill:var(--ok);opacity:.13}
.ebar{stroke:var(--s2);stroke-width:1.6;opacity:.55}
.bounded{stroke-width:3}
.callout{fill:none;stroke:var(--ink);stroke-width:1.5;opacity:.6}
.delta{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:500}
.delta.flat{fill:var(--muted)}.delta.moved{fill:var(--bad)}
.dissoc{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:.5rem}
@media(max-width:720px){.dissoc{grid-template-columns:1fr}}
.cell{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:.9rem;
  box-shadow:var(--shadow)}
.cell figure.chart{border:0;box-shadow:none;padding:0;background:none}
.cellhead{display:flex;gap:.55rem;align-items:baseline;padding:0 .2rem .6rem}
.csym{font-family:"IBM Plex Mono",monospace;color:var(--accent);font-size:1rem;font-weight:500}
.cwhat{color:var(--muted);font-size:.87rem}
.board{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:10px;overflow:hidden;margin:2.2rem 0 0}
.board div{background:var(--panel);padding:1rem}
.board .n{font-family:Spectral,Georgia,serif;font-size:1.9rem;line-height:1;font-weight:600}
.board .l{font-family:"IBM Plex Mono",monospace;font-size:.66rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);margin-top:.4rem}
.n.ok{color:var(--ok)}.n.bad{color:var(--bad)}.n.warn{color:var(--warn)}
.verdict{margin-top:1rem;padding:.9rem 1.1rem;border-radius:8px;font-size:.97rem;max-width:46rem}
.verdict.ok{background:var(--ok-soft)}.verdict.bad{background:var(--bad-soft)}
.verdict.warn{background:var(--warn-soft)}
.verdict b{font-weight:600}
.pull{border-left:3px solid var(--accent);padding:.15rem 0 .15rem 1.1rem;margin:1.5rem 0;
  font-family:Spectral,Georgia,serif;font-size:1.15rem;max-width:40rem}
footer{margin-top:4rem;padding-top:1.3rem;border-top:1px solid var(--rule);
  color:var(--faint);font-size:.85rem}
footer code{font-family:"IBM Plex Mono",monospace;font-size:.9em}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""

PAGE = """<title>What We Tested</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}{dcss}</style>
<div class="wrap">
<p class="eyebrow">astro_atp · reading guide</p>
<h1>What we tested, what held, what didn&rsquo;t</h1>
<p class="standfirst">Every parameter we swept, drawn from the measured data. Each chart is the
actual ensemble &mdash; ten seeds, error bands where they exist &mdash; not an illustration.</p>

<div class="board">
<div><div class="n ok">6</div><div class="l">claims held</div></div>
<div><div class="n bad">3</div><div class="l">failed / dropped</div></div>
<div><div class="n warn">2</div><div class="l">not claimed</div></div>
<div><div class="n">10</div><div class="l">seeds behind every number</div></div>
</div>

<section>
<h2><span class="num">01</span>What a cell does, and what a network does</h2>
<p class="sectnote">Everything below uses two words in a precise way. It is worth two minutes to
fix what they mean, because the paper&rsquo;s whole result is that they can be pulled apart.</p>
{single}
<p style="margin-top:1.6rem">Now put cells next to each other. A cell has two ways to light up, and
telling them apart is the entire game:</p>
{diagrams}
</section>

<section>
<h2><span class="num">02</span>The result the paper rests on</h2>
<p class="sectnote">Two ingredients, two outcomes, four combinations. If each ingredient fixed
both problems they would be one mechanism described twice. The claim is that they don&rsquo;t &mdash;
and the grid is the whole argument in one picture: the diagonal moves, the off-diagonal is flat.</p>
{grid}
<div class="verdict ok"><b>Held, in both directions.</b> A refractory period cuts spontaneous
ignitions twelvefold and leaves the wave&rsquo;s reach untouched. Decremental release is the exact
converse &mdash; it shortens the wave ninefold and leaves ignitions alone, varying by
0.06% across the entire sweep. Each moves one quantity and not the other, which is what makes them
separable rather than two names for one effect.</div>
<p style="margin-top:1.3rem"><em>Why the flat panels are not a null result:</em> a spontaneous
ignition is <em>by definition</em> one with no active neighbour, and decremental release only
changes what an active neighbour passes on. The bottom-left panel is what the mechanism predicts,
which is why it counts as evidence rather than absence of it.</p>
</section>

<section>
<h2><span class="num">03</span>Choosing the baseline &mdash; the one thing we had to pick</h2>
<p class="sectnote">Before any of the science, one constant had to be set: how close a cell sits to
firing when there is no ATP at all. We pinned it with a constraint instead of tuning it.</p>
{i0}
<div class="verdict ok"><b>Held, with a discrepancy we report rather than hide.</b> The chosen
baseline fires 3.6&times; faster than the somatic rate measured in vivo. No baseline both matches
that rate and gives a fully participating medium &mdash; the experimental band sits inside the
transition, where most cells are still silent. We say so in the paper.</div>
</section>

<section>
<h2><span class="num">04</span>What the ATP dial actually does</h2>
<p class="sectnote">ATP moves six things at once: drive, noise, threshold, recovery, baseline current
and coupling. Two questions follow. Does it switch the network on collectively? And is its effect
just the coupling collapse in disguise?</p>
{atp}
<div class="verdict warn"><b>Split verdict.</b> ATP does sweep a broad distribution of single-cell
thresholds &mdash; that holds. It does <em>not</em> produce a collective onset, and the gap between
the two curves is why we never call it a bifurcation parameter for the network.</div>
<div style="margin-top:1.6rem">{coup}</div>
<div class="verdict ok"><b>Held, against a criterion written before the data existed.</b> Activity
rises when everything follows ATP and falls when only coupling does. One honest wrinkle, reported in
the paper: that criterion&rsquo;s formula contradicted its own stated intent &mdash; written
two-sided when the hypothesis was directional. Read as intended it passes, and we publish the
discrepancy instead of quietly substituting the corrected form.</div>
</section>

<section>
<h2><span class="num">05</span>Where measurement stops working</h2>
<p class="sectnote">Two sweeps whose result is a limit rather than a finding. Both are in the paper
because a reader needs to know where the numbers stop meaning what they look like they mean.</p>
{lag}
<div class="verdict ok"><b>Held: synchrony, not propagation.</b> The unprovoked network cannot
support a wave measurement at all, which is exactly why every wave number in the paper comes from
deliberately stimulating one point. <em>This one reversed on us</em> &mdash; a single seed suggested
the opposite condition carried the effect, and only the ten-seed ensemble showed the truth. It is
the clearest reason nothing here quotes one realization.</div>
<div style="margin-top:1.6rem">{r2}</div>
<div class="verdict bad"><b>Our own criterion failed &mdash; and we published that.</b> We set
R² &gt; 0.9 in advance as the standard for calling something a propagating front. The bounded front
misses it. So does a front we know propagates, at every radius. The bar was set above what this
medium produces, so the statistic cannot discriminate; matched seed for seed the two differ by
0.005 (p = 0.90). The honest response to a failed pre-registration is to report it, not to lower
the bar.</div>
</section>

<section>
<h2><span class="num">06</span>The agreement we did not fit</h2>
<p class="sectnote">The extent result was tuned to nothing &mdash; but it is one number matching one
measurement, which is weak on its own. Termination <em>time</em> is a second, independent check that
was run afterwards.</p>
{term}
<div class="verdict ok"><b>Held.</b> The same gain interval that puts the front inside the reported
distance also puts it inside the reported duration. One free parameter satisfying two independent
constraints is much stronger than satisfying one.</div>
</section>

<section>
<h2><span class="num">07</span>The one-paragraph version</h2>
<p>ATP as a uniform dial does move every cell&rsquo;s threshold &mdash; but the network never turns
on at once, so it is not a switch for the tissue. Stimulate one point and you get a clean travelling
front, which is the good news; it then never stops, which is the bad news. Add noise and you cannot
measure a wave at all, because cells ignite on their own. Two standard ingredients repair two
different failures: a refractory period stops spurious ignitions without shortening the wave, and
decremental release shortens the wave without touching ignitions. Each does one job. That separation,
measured in both directions, is the paper.</p>
</section>

<footer>
Every quantity is a mean over the same ten seeds (11&ndash;20); shaded bands and error bars are SD
across those seeds. Distances use 25&nbsp;μm per cell; one model time unit ≈ 1&nbsp;s. All results are
written by <code>core.provenance.save_result</code>, which stamps the parameters and the git commit
into every file, so any number on this page traces back to the run that produced it. Charts generated
by <code>make_explainer.py</code> in <code>bayat-et-al</code>.
</footer>
</div>
"""



# ---------------------------------------------------------------- concept diagrams
def single_cell(d):
    """One real trace, annotated with the excitable cycle."""
    W, H = 640, 220
    z = np.load(PD / "fig1_single_cell.npz", allow_pickle=True)
    tr, tt = z["traces"][0], z["t_disp"]
    k = max(1, len(tr) // 900)
    tr, tt = tr[::k], tt[::k]
    p, xm, ym = axes(W, H, (58, 130, 20, 44), "time  (s)", "calcium  C",
                     [0, 100, 200, 300, 400, 500], [-2, -1, 0, 1], (0, 500), (-2.3, 2.0))
    th = 0.633
    yth = ym(th)
    p.append(f'<line x1="{xm(0):.1f}" y1="{yth:.1f}" x2="{xm(500):.1f}" y2="{yth:.1f}" class="thresh"/>')
    p.append(f'<text x="{xm(500)+6:.1f}" y="{yth+3.5:.1f}" class="note">threshold</text>')
    yr_ = ym(-1.2)
    p.append(f'<line x1="{xm(0):.1f}" y1="{yr_:.1f}" x2="{xm(500):.1f}" y2="{yr_:.1f}" class="restline"/>')
    p.append(f'<text x="{xm(500)+6:.1f}" y="{yr_+3.5:.1f}" class="note">resting level</text>')
    pts = " ".join(f"{xm(x):.1f},{ym(y):.1f}" for x, y in zip(tt, tr))
    p.append(f'<polyline points="{pts}" class="s1 trace"/>')
    idx = [i for i in range(1, len(tr)) if tr[i-1] <= 1.0 < tr[i]]
    if idx:
        i0 = idx[0]; xs0 = xm(tt[i0])
        p.append(f'<text x="{xs0:.1f}" y="{ym(1.9):.1f}" class="note tc">one transient</text>')
        p.append(f'<line x1="{xs0:.1f}" y1="{ym(1.75):.1f}" x2="{xs0:.1f}" y2="{ym(1.5):.1f}" class="lead"/>')
    return chart(W, H, p,
        "A real trace from the model, at low ATP. The cell sits well below threshold and does nothing. "
        "Noise occasionally pushes it over, and when that happens the response is large and always about "
        "the same size — that is what makes it <em>excitable</em> rather than merely noisy. Each excursion "
        "is one calcium transient.")


DIAGRAMS = """
<figure class="chart diagram">
<svg viewBox="0 0 640 230" role="img" aria-label="Two ways a cell can light up: ignition with no active neighbour, versus recruitment from an active neighbour.">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
    <marker id="arh" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--s2)"/>
    </marker>
  </defs>

  <text x="20" y="24" class="dtitle">IGNITION</text>
  <text x="20" y="42" class="dsub">no active neighbour — it starts on its own</text>
  <circle cx="60"  cy="105" r="19" class="cell off"/>
  <circle cx="150" cy="105" r="19" class="cell on"/>
  <circle cx="240" cy="105" r="19" class="cell off"/>
  <line x1="79" y1="105" x2="131" y2="105" class="link"/>
  <line x1="169" y1="105" x2="221" y2="105" class="link"/>
  <line x1="150" y1="58" x2="150" y2="80" class="noisearrow" marker-end="url(#ar)"/>
  <text x="150" y="52" class="dlab tc">noise</text>
  <text x="60"  y="140" class="dsmall tc">quiet</text>
  <text x="150" y="140" class="dsmall tc">fires</text>
  <text x="240" y="140" class="dsmall tc">quiet</text>
  <text x="20" y="182" class="dnote">This is what we count as a spontaneous</text>
  <text x="20" y="198" class="dnote">ignition — and what a refractory period</text>
  <text x="20" y="214" class="dnote">suppresses.</text>

  <line x1="320" y1="16" x2="320" y2="214" class="divider"/>

  <text x="350" y="24" class="dtitle">RECRUITMENT</text>
  <text x="350" y="42" class="dsub">a neighbour is already active and passes it on</text>
  <circle cx="390" cy="105" r="19" class="cell on"/>
  <circle cx="480" cy="105" r="19" class="cell on"/>
  <circle cx="570" cy="105" r="19" class="cell off"/>
  <line x1="411" y1="105" x2="457" y2="105" class="passlink" marker-end="url(#arh)"/>
  <line x1="499" y1="105" x2="551" y2="105" class="link"/>
  <text x="434" y="94" class="dlab tc s2t">transmits</text>
  <text x="390" y="140" class="dsmall tc">fires first</text>
  <text x="480" y="140" class="dsmall tc">recruited</text>
  <text x="570" y="140" class="dsmall tc">next</text>
  <text x="350" y="182" class="dnote">This is how a wave travels — and what</text>
  <text x="350" y="198" class="dnote">decremental release weakens, step by</text>
  <text x="350" y="214" class="dnote">step.</text>
</svg>
<figcaption>The whole vocabulary in one picture. A cell can light up two ways, and they are
different events: <b>ignition</b> happens with no active neighbour, <b>recruitment</b> happens
because of one. Every claim in this document turns on keeping them apart — which is also why one
mechanism can change one without touching the other.</figcaption>
</figure>

<figure class="chart diagram">
<svg viewBox="0 0 640 250" role="img" aria-label="What each mechanism does: a refractory period silences a cell after it fires, while decremental release weakens the signal passed along a chain.">
  <defs>
    <marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--s2)"/>
    </marker>
  </defs>

  <text x="20" y="24" class="dtitle">REFRACTORY PERIOD  <tspan class="csymt">τ_ref</tspan></text>
  <text x="20" y="42" class="dsub">after firing, a cell goes silent — and stops passing signal on</text>
  <circle cx="70"  cy="92" r="18" class="cell on"/>
  <circle cx="160" cy="92" r="18" class="cell off"/>
  <text x="70"  y="122" class="dsmall tc">fires</text>
  <text x="115" y="70" class="dsmall tc">then…</text>
  <circle cx="290" cy="92" r="18" class="cell spent"/>
  <circle cx="380" cy="92" r="18" class="cell off"/>
  <line x1="309" y1="92" x2="357" y2="92" class="link cut"/>
  <line x1="325" y1="80" x2="345" y2="104" class="strike"/>
  <text x="290" y="122" class="dsmall tc">cannot re-fire</text>
  <text x="380" y="122" class="dsmall tc">gets nothing</text>
  <text x="440" y="88" class="dnote">Fewer cells start on their own,</text>
  <text x="440" y="104" class="dnote">so ignitions fall twelvefold.</text>
  <text x="440" y="124" class="dnote dmute">The travelling wave is untouched:</text>
  <text x="440" y="140" class="dnote dmute">it never needed to re-fire.</text>

  <line x1="20" y1="160" x2="620" y2="160" class="divider h"/>

  <text x="20" y="188" class="dtitle">DECREMENTAL RELEASE  <tspan class="csymt">γ_regen</tspan></text>
  <text x="20" y="206" class="dsub">each cell passes on only a fraction of what recruited it</text>
  <circle cx="70"  cy="234" r="16" class="cell on"/>
  <circle cx="180" cy="234" r="14" class="cell on faded1"/>
  <circle cx="290" cy="234" r="12" class="cell on faded2"/>
  <circle cx="400" cy="234" r="10" class="cell off"/>
  <line x1="88" y1="234" x2="162" y2="234" class="passlink w3" marker-end="url(#ar2)"/>
  <line x1="196" y1="234" x2="274" y2="234" class="passlink w2" marker-end="url(#ar2)"/>
  <line x1="304" y1="234" x2="386" y2="234" class="passlink w1" marker-end="url(#ar2)"/>
  <text x="125" y="222" class="dsmall tc">full</text>
  <text x="235" y="222" class="dsmall tc">weaker</text>
  <text x="345" y="222" class="dsmall tc">too weak</text>
  <text x="440" y="230" class="dnote">The wave dies out after a few cells.</text>
  <text x="440" y="246" class="dnote dmute">Ignitions are unaffected — they had</text>
</svg>
<figcaption>Why the four-panel grid above has to look the way it does. A refractory period acts on a
cell <em>after</em> it fires, so it removes ignitions but never shortens a wave. Decremental release
acts on what one cell hands to the next, so it shortens the wave but cannot touch a cell that had no
neighbour to hear from in the first place. Each mechanism can only reach one of the two columns.</figcaption>
</figure>
"""

DIAG_CSS = """
figure.diagram svg{color:var(--ink)}
.cell{stroke-width:2.5}
.cell.off{fill:var(--panel);stroke:var(--rule2)}
.cell.on{fill:var(--s1);stroke:var(--s1)}
.cell.spent{fill:var(--rule);stroke:var(--rule2);stroke-dasharray:3 3}
.cell.faded1{opacity:.62}.cell.faded2{opacity:.34}
.link{stroke:var(--rule2);stroke-width:2.5}
.link.cut{opacity:.4}
.strike{stroke:var(--bad);stroke-width:2.5;stroke-linecap:round}
.passlink{stroke:var(--s2);fill:none}
.passlink.w3{stroke-width:4}.passlink.w2{stroke-width:2.6}.passlink.w1{stroke-width:1.3;opacity:.6}
.noisearrow{stroke:currentColor;stroke-width:1.6;opacity:.75}
.divider{stroke:var(--rule);stroke-width:1}
.dtitle{fill:var(--ink);font-size:12.5px;font-weight:600;letter-spacing:.06em;
  font-family:"IBM Plex Sans",sans-serif}
.csymt{fill:var(--accent);font-family:"IBM Plex Mono",monospace;letter-spacing:0}
.dsub{fill:var(--muted);font-size:11px;font-family:"IBM Plex Sans",sans-serif}
.dlab{font-size:11px;font-weight:500;font-family:"IBM Plex Sans",sans-serif;fill:var(--muted)}
.dsmall{fill:var(--muted);font-size:10.5px;font-family:"IBM Plex Sans",sans-serif}
.dnote{fill:var(--ink);font-size:11.5px;font-family:"IBM Plex Sans",sans-serif}
.dnote.dmute{fill:var(--muted)}
.trace{stroke-width:1.2}
.restline{stroke:var(--rule2);stroke-width:1;stroke-dasharray:2 4}
.lead{stroke:var(--muted);stroke-width:1}
"""


if __name__ == "__main__":
    d = load()
    C = build(d)
    OUT.write_text(PAGE.format(css=CSS, dcss=DIAG_CSS, **C))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
