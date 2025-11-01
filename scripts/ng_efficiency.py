import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.sankey import Sankey

# =========================
# Config / constants
# =========================
# LHV (MJ/kmol)
LHV = {
    'CH4': 802.3,
    'C2H6': 1428.8,
    'C3H8': 2043.9,
    'iC4_nC4_C4': 2658.0,
    'CO': 283.0,
    'H2': 241.8
}
components_for_calc = list(LHV.keys())
COMPONENTS_FEEDLIKE = ['CH4', 'C2H6', 'C3H8', 'iC4_nC4_C4', 'CO']  # feed/fuel contributors

# =========================
# Load & clean
# =========================
df = pd.read_excel('./data/NG_EOR_Heat_Material_Balance.xlsx')
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
df.dropna(how='all', inplace=True)
df['Enthalpy_Flow_kW_num'] = pd.to_numeric(df.get('Enthalpy_Flow_kW', 0), errors='coerce').fillna(0)
df['Molar_Flow_kmol_per_h'] = pd.to_numeric(df.get('Molar_Flow_kmol_per_h', 0), errors='coerce').fillna(0)

# =========================
# Helper calcs
# =========================
def compute_stream_energy(row, components):
    """Sum (xi * n * LHV_i) across listed components."""
    if pd.isna(row['Molar_Flow_kmol_per_h']):
        return 0.0
    total = 0.0
    for comp in components:
        if pd.notna(row.get(comp)) and comp in LHV:
            x = row[comp]
            n = row['Molar_Flow_kmol_per_h']
            total += x * n * LHV[comp]
    return total

def component_energy_breakdown(df_streams, components):
    """Return DataFrame with per-component energy (MJ/h) summed across streams."""
    rows = []
    for _, r in df_streams.iterrows():
        n = r['Molar_Flow_kmol_per_h']
        for comp in components:
            if pd.notna(r.get(comp)) and comp in LHV:
                x = r[comp]
                rows.append({
                    "stream": r['Description'],
                    "component": comp,
                    "energy_MJph": x * n * LHV[comp]
                })
    if not rows:
        return pd.DataFrame(columns=["stream", "component", "energy_MJph"])
    out = pd.DataFrame(rows)
    return out

def savefig(name):
    plt.tight_layout()
    out = PLOTS_DIR / name
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out}")

# =========================
# Identify streams
# =========================
feedstock_streams = df[df['Description'].str.contains('Natural Gas|NG', case=False, na=False)]
fuel_streams      = df[df['Description'].str.contains('Fuel Gas', case=False, na=False)]
product_streams   = df[df['Description'].str.contains('Hydrogen Product', case=False, na=False)]
steam_inputs      = df[df['Description'].str.contains('Steam to', case=False, na=False)]
steam_exports     = df[df['Description'].str.contains('Steam Export', case=False, na=False)]
tailgas_streams   = df[df['Description'].str.contains('Tail Gas', case=False, na=False)]

# =========================
# Energies (MJ/h)
# =========================
feedstock_energy = feedstock_streams.apply(compute_stream_energy, components=components_for_calc, axis=1).sum()
fuel_energy      = fuel_streams.apply(compute_stream_energy, components=components_for_calc, axis=1).sum()
product_energy   = product_streams.apply(compute_stream_energy, components=['H2'], axis=1).sum()
tailgas_energy   = tailgas_streams.apply(compute_stream_energy, components=components_for_calc, axis=1).sum()

steam_in_MJ  = abs(steam_inputs['Enthalpy_Flow_kW_num'].sum())  * 3.6
steam_out_MJ = abs(steam_exports['Enthalpy_Flow_kW_num'].sum()) * 3.6

# Fired heater extra loss ~12% of fired fuel
heat_loss_fraction = 0.12
heat_losses_extra = fuel_energy * heat_loss_fraction
print(f"(Estimated fired-heater losses: {heat_losses_extra:.2f} MJ/h, ≈{heat_loss_fraction*100:.0f}% of fuel energy)")

# Total input
total_input_energy = feedstock_energy + fuel_energy + steam_in_MJ + heat_losses_extra

# Other cooling/heat-loss enthalpy (from negatives)
heat_losses_kW = df[df['Enthalpy_Flow_kW_num'] < 0]['Enthalpy_Flow_kW_num'].sum()
total_heat_losses = abs(heat_losses_kW) * 3.6

# Useful outputs
total_useful_output = product_energy + steam_out_MJ

# =========================
# Textual report
# =========================
print("\n===============================")
print("NG EOR – ENERGY BALANCE")
print("===============================")
print(f"Feedstock: {feedstock_energy:,.2f} MJ/h")
print(f"Fuel gas:  {fuel_energy:,.2f} MJ/h")
print(f"Steam in:  {steam_in_MJ:,.2f} MJ/h")
print(f"Furnace losses (est.):  {heat_losses_extra:,.2f} MJ/h")
print(f"TOTAL INPUT: {total_input_energy:,.2f} MJ/h\n")

print(f"H₂ product: {product_energy:,.2f} MJ/h")
print(f"Steam export: {steam_out_MJ:,.2f} MJ/h")
print(f"Tail gas: {tailgas_energy:,.2f} MJ/h")
print(f"Heat losses (cooling etc.): {total_heat_losses:,.2f} MJ/h")

# Efficiencies
eff_h2 = (product_energy / total_input_energy) * 100 if total_input_energy > 0 else 0.0
eff_with_steam = ( (product_energy + steam_out_MJ) / total_input_energy ) * 100 if total_input_energy > 0 else 0.0
chem_inputs_only = feedstock_energy + fuel_energy
eff_chem = (product_energy / chem_inputs_only) * 100 if chem_inputs_only > 0 else 0.0
eff_feed = (product_energy / feedstock_energy) * 100 if feedstock_energy > 0 else 0.0

print("\n--- EFFICIENCIES ---")
print(f"Thermal Efficiency (H₂ only): {eff_h2:.2f}%")
print(f"Thermal Efficiency (H₂ + steam export): {eff_with_steam:.2f}%")
print(f"Chemical Efficiency (H₂ / chemical inputs): {eff_chem:.2f}%")
print(f"Feedstock Conversion Efficiency: {eff_feed:.2f}%")
