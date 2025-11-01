from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
OUT_DIR = Path("data/efficiency_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Files (adjust if needed)
FILE_LPG = Path("data/LPG_EOR_Heat_Material_Balance.xlsx")
FILE_NG  = Path("data/NG_EOR_Heat_Material_Balance.xlsx")
SHEET    = "All_Streams"  # if missing, we'll default to first sheet

# Lower Heating Values, MJ/kmol
LHV = {
    "CH4": 802.3,
    "C2H6": 1428.8,
    "C3H8": 2043.9,
    "iC4_nC4_C4": 2658.0,
    "CO": 283.0,
    "H2": 241.8,
}
COMP_FEEDLIKE = ["CH4", "C2H6", "C3H8", "iC4_nC4_C4", "CO"]
FURNACE_LOSS_FRAC = 0.12  # ≈12% of fired fuel (common SMR rule-of-thumb)

# -----------------------------
# Helpers
# -----------------------------
def _load_xls(path: Path) -> pd.DataFrame:
    """Load and normalize columns; if sheet not present, use first."""
    try:
        df = pd.read_excel(path, sheet_name=SHEET)
    except Exception:
        df = pd.read_excel(path)  # first sheet fallback
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    df.dropna(how="all", inplace=True)
    # numeric helpers
    df["Enthalpy_Flow_kW_num"]    = pd.to_numeric(df.get("Enthalpy_Flow_kW", 0), errors="coerce").fillna(0)
    df["Molar_Flow_kmol_per_h"]   = pd.to_numeric(df.get("Molar_Flow_kmol_per_h", 0), errors="coerce").fillna(0)
    return df

def _energy_stream(row, components):
    """Σ (x_i * n * LHV_i) for a single row (MJ/h)."""
    n = float(row.get("Molar_Flow_kmol_per_h", 0))
    if not np.isfinite(n) or n == 0:
        return 0.0
    total = 0.0
    for c in components:
        if c in LHV and pd.notna(row.get(c)):
            x = float(row[c])
            total += x * n * LHV[c]
    return total

def _sum_energy(df, components):
    return df.apply(_energy_stream, components=components, axis=1).sum()

def _component_stack(df, comps):
    rows = []
    for _, r in df.iterrows():
        n = float(r.get("Molar_Flow_kmol_per_h", 0))
        for c in comps:
            x = r.get(c, np.nan)
            if pd.notna(x):
                rows.append({"component": c, "E": x * n * LHV[c]})
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.Series({c: 0.0 for c in comps})
    return d.groupby("component")["E"].sum().reindex(comps, fill_value=0.0)

def _analyze_case(tag: str, path: Path):
    """
    Compute a full balance for a case.
    Returns a dict with inputs, outputs, losses, component stacks and efficiencies.
    """
    df = _load_xls(path)

    # Stream groups
    if tag.lower().startswith("lpg"):
        feed_pat = "LPG|Natural Gas"
        feedname = "Feedstock (LPG/NG)"
    else:
        feed_pat = "Natural Gas|NG"
        feedname = "Feedstock (NG)"

    feed_df    = df[df["Description"].str.contains(feed_pat, case=False, na=False)]
    fuel_df    = df[df["Description"].str.contains("Fuel Gas", case=False, na=False)]
    prod_df    = df[df["Description"].str.contains("Hydrogen Product", case=False, na=False)]
    steam_in   = df[df["Description"].str.contains("Steam to", case=False, na=False)]
    steam_out  = df[df["Description"].str.contains("Steam Export", case=False, na=False)]
    tailgas_df = df[df["Description"].str.contains("Tail Gas", case=False, na=False)]

    # Energies (MJ/h)
    E_feed   = _sum_energy(feed_df, list(LHV.keys()))
    E_fuel   = _sum_energy(fuel_df, list(LHV.keys()))
    E_H2     = _sum_energy(prod_df, ["H2"])
    E_tail   = _sum_energy(tailgas_df, list(LHV.keys()))
    E_steam_in   = abs(steam_in["Enthalpy_Flow_kW_num"].sum()) * 3.6
    E_steam_out  = abs(steam_out["Enthalpy_Flow_kW_num"].sum()) * 3.6
    # Electricity: none (per your data)
    E_electric   = 0.0

    # Furnace loss ~12% of fired fuel
    E_furnace_loss = E_fuel * FURNACE_LOSS_FRAC

    # Total inputs for efficiency denominator
    E_inputs = E_feed + E_fuel + E_steam_in + E_furnace_loss + E_electric

    # Cooling/other negative enthalpy (diagnostic)
    heat_losses_kW = df[df["Enthalpy_Flow_kW_num"] < 0]["Enthalpy_Flow_kW_num"].sum()
    E_cooling_loss = abs(heat_losses_kW) * 3.6

    # Efficiencies
    eff_H2_only      = (E_H2 / E_inputs) * 100 if E_inputs > 0 else 0.0
    eff_H2_withSteam = ((E_H2 + E_steam_out) / E_inputs) * 100 if E_inputs > 0 else 0.0
    chem_inputs_only = E_feed + E_fuel
    eff_chem_only    = (E_H2 / chem_inputs_only) * 100 if chem_inputs_only > 0 else 0.0
    eff_feed_conv    = (E_H2 / E_feed) * 100 if E_feed > 0 else 0.0

    # Component stacks (feed & fuel)
    stack_feed = _component_stack(feed_df, COMP_FEEDLIKE)
    stack_fuel = _component_stack(fuel_df, COMP_FEEDLIKE)

    return dict(
        tag=tag,
        inputs=dict(**{feedname: E_feed, "Fuel Gas": E_fuel, "Steam Input": E_steam_in}),
        outputs=dict(**{
            "H₂ Product": E_H2,
            "Steam Export": E_steam_out,
            "Tail Gas": E_tail,
            "Furnace Loss": E_furnace_loss,
            "Cooling Loss": E_cooling_loss,  # diagnostic (not in efficiency denominator)
        }),
        feed_stack=stack_feed,
        fuel_stack=stack_fuel,
        efficiencies=dict(
            H2_only=eff_H2_only,
            H2_plus_steam=eff_H2_withSteam,
            Chem_only=eff_chem_only,
            Feedstock_conv=eff_feed_conv,
        )
    )

def savefig(name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    p = OUT_DIR / name
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[saved] {p}")

def _fmt(x): return f"{x:,.0f}"

# -----------------------------
# Build both cases
# -----------------------------
lpg = _analyze_case("LPG_EOR", FILE_LPG)
ng  = _analyze_case("NG_EOR",  FILE_NG)

# -----------------------------
# 1) Inputs comparison (grouped bars)
# -----------------------------
cats = ["Feedstock", "Fuel Gas", "Steam Input"]
# Normalize feed label difference
def _pull_inputs(d):
    keys = list(d["inputs"].keys())
    feed_key = [k for k in keys if k.startswith("Feedstock")][0]
    return [
        d["inputs"][feed_key],
        d["inputs"]["Fuel Gas"],
        d["inputs"]["Steam Input"]
    ]

vals_lpg = _pull_inputs(lpg)
vals_ng  = _pull_inputs(ng)

x = np.arange(len(cats))
w = 0.38
plt.figure(figsize=(9,4))
b1 = plt.bar(x - w/2, vals_lpg, width=w, label="LPG", color="#1f77b4")
b2 = plt.bar(x + w/2, vals_ng,  width=w, label="NG",  color="#ff7f0e")
plt.xticks(x, cats)
plt.ylabel("Energy (MJ/h)")
plt.title("Inputs Comparison (Σ(x·n·LHV) + Steam + FurnaceLoss(≈12% Fuel))")
for b in list(b1)+list(b2):
    plt.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, _fmt(b.get_height()), ha='center', fontsize=8)
plt.legend()
savefig("01_inputs_comparison.png")
# -----------------------------
# 2) Outputs & losses comparison
# -----------------------------
oc = ["H₂ Product", "Steam Export", "Tail Gas", "Furnace Loss", "Cooling Loss"]
vl = [lpg["outputs"][k] for k in oc]
vn = [ng["outputs"][k]  for k in oc]

x = np.arange(len(oc))
plt.figure(figsize=(11,4))
b1 = plt.bar(x - w/2, vl, width=w, label="LPG", color="#2ca02c")
b2 = plt.bar(x + w/2, vn, width=w, label="NG",  color="#d62728")
plt.xticks(x, oc, rotation=15, ha='right')
plt.ylabel("Energy (MJ/h)")
plt.title("Outputs & Losses Comparison")
for b in list(b1)+list(b2):
    plt.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, _fmt(b.get_height()), ha='center', fontsize=8)
plt.legend()
savefig("02_outputs_losses_comparison.png")

# -----------------------------
# 3) Feedstock component stacks (LPG vs NG)
# -----------------------------
comp = COMP_FEEDLIKE
lpg_feed = lpg["feed_stack"].values
ng_feed  = ng["feed_stack"].values

plt.figure(figsize=(10,5))
bottom = np.zeros(2)
for i, c in enumerate(comp):
    vals = np.array([lpg["feed_stack"][c], ng["feed_stack"][c]])
    plt.bar(["LPG", "NG"], vals, bottom=bottom, label=c)
    bottom += vals
plt.ylabel("Energy (MJ/h)")
plt.title("Feedstock Energy by Component (Σ x·n·LHV per component)")
plt.legend(title="Component", bbox_to_anchor=(1.01,1), loc="upper left", frameon=False)
for i,val in enumerate(bottom):
    plt.text(i, val*1.01, _fmt(val), ha='center', fontsize=9)
savefig("03_feed_component_stacks.png")

# -----------------------------
# 4) Fuel gas component stacks (LPG vs NG)
# -----------------------------
plt.figure(figsize=(10,5))
bottom = np.zeros(2)
for c in comp:
    vals = np.array([lpg["fuel_stack"][c], ng["fuel_stack"][c]])
    plt.bar(["LPG", "NG"], vals, bottom=bottom, label=c)
    bottom += vals
plt.ylabel("Energy (MJ/h)")
plt.title("Fuel Gas Energy by Component (Σ x·n·LHV per component)")
plt.legend(title="Component", bbox_to_anchor=(1.01,1), loc="upper left", frameon=False)
for i,val in enumerate(bottom):
    plt.text(i, val*1.01, _fmt(val), ha='center', fontsize=9)
savefig("04_fuel_component_stacks.png")

# -----------------------------
# 5) Efficiency comparison (four metrics)
# -----------------------------
metrics = [
    ("H₂-only",          "H2_only"),
    ("H₂ + Steam",       "H2_plus_steam"),
    ("Chemical only",    "Chem_only"),
    ("Feedstock conv.",  "Feedstock_conv"),
]
labels = [m[0] for m in metrics]
lpg_vals = [lpg["efficiencies"][m[1]] for m in metrics]
ng_vals  = [ng["efficiencies"][m[1]]  for m in metrics]

x = np.arange(len(labels))
plt.figure(figsize=(10,4))
b1 = plt.bar(x - w/2, lpg_vals, width=w, label="LPG", color="#1f77b4")
b2 = plt.bar(x + w/2, ng_vals,  width=w, label="NG",  color="#ff7f0e")
plt.xticks(x, labels, rotation=0)
plt.ylabel("Efficiency (%)")
plt.title("Efficiency Comparison\n"
          "H₂-only = E_H2 / Inputs;  H₂+Steam = (E_H2 + E_steamExport) / Inputs;\n"
          "Chemical only = E_H2 / (E_feed + E_fuel);  Feedstock conv. = E_H2 / E_feed")
for b in list(b1)+list(b2):
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.1f}%", ha='center', fontsize=8)
plt.ylim(0, max(lpg_vals+ng_vals)*1.25 + 5)
plt.legend()
savefig("05_efficiency_comparison.png")

# -----------------------------
# 6) (Optional) two-panel waterfall (LPG vs NG)
# -----------------------------
def _waterfall(ax, case_dict, title):
    # Normalize feed key
    feed_key = [k for k in case_dict["inputs"].keys() if k.startswith("Feedstock")][0]
    steps = [
        (feed_key, case_dict["inputs"][feed_key]),         # + feed
        ("Fuel Gas", case_dict["inputs"]["Fuel Gas"]),     # + fuel
        ("Steam Input", case_dict["inputs"]["Steam Input"]),  # + steam
        ("H₂ Product", -case_dict["outputs"]["H₂ Product"]),
        ("Steam Export", -case_dict["outputs"]["Steam Export"]),
        ("Tail Gas", -case_dict["outputs"]["Tail Gas"]),
        ("Furnace Loss", -case_dict["outputs"]["Furnace Loss"]),
    ]
    running = 0.0
    xs, ys, heights = [], [], []
    for i,(lab,val) in enumerate(steps):
        base = running if val>=0 else running+val
        xs.append(i); ys.append(base); heights.append(abs(val))
        running += val
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#17becf","#9467bd","#d62728","#bcbd22"]
    for i,(x0,y0,h) in enumerate(zip(xs,ys,heights)):
        ax.bar(x0, h, bottom=y0, color=colors[i%len(colors)])
        val = steps[i][1]
        ax.text(x0, y0 + (h*1.02 if val>=0 else -h*0.02), _fmt(val), ha='center', va='bottom' if val>=0 else 'top', fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], rotation=25, ha='right')
    ax.set_ylabel("Energy (MJ/h)")
    ax.set_title(title)

plt.figure(figsize=(13,5))
ax1 = plt.subplot(1,2,1)
_waterfall(ax1, lpg, "LPG – Energy Waterfall\nInputs positive, outputs/losses negative")
ax2 = plt.subplot(1,2,2)
_waterfall(ax2, ng,  "NG – Energy Waterfall\nInputs positive, outputs/losses negative")
plt.suptitle("Waterfall (Formula: Inputs = Σ(x·n·LHV) + Steam(kW·3.6) + FurnaceLoss(≈12% Fuel))", y=1.02, fontsize=11)
plt.tight_layout()
savefig("06_waterfall_lpg_vs_ng.png")

print(f"\nAll plots saved to: {OUT_DIR.resolve()}")
