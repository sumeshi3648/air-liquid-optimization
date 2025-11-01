import pandas as pd

HHV = {
    'CH4': 891.0,
    'C2H6': 1560.6,
    'C3H8': 2222.0,
    'iC4_nC4_C4': 2875.0,
    'CO': 283.2,
    'H2': 286.0
}

components_for_calc = list(HHV.keys())  # Use these directly

# Load stream data
df = pd.read_excel('./data/LPG_EOR_Heat_Material_Balance.xlsx', sheet_name='All_Streams')
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
df.dropna(how='all', inplace=True)


def compute_stream_energy(row, components):
    if pd.isna(row['Molar_Flow_kmol_per_h']):
        return 0
    energy = 0
    for comp in components:
        if pd.notna(row.get(comp)) and comp in HHV:
            x = row[comp]
            n = row['Molar_Flow_kmol_per_h']
            energy += x * n * HHV[comp]
    return energy


# --- Identify feedstock, fuel, and product streams ---
# LPG is the primary feedstock that gets converted to hydrogen
# Natural Gas might be additional feedstock (if it has flow)
# Fuel Gas is tail gas that can be used as fuel
feedstock_keywords = ['LPG', 'Natural Gas']  # Primary feedstock for hydrogen production
fuel_keywords = ['Fuel Gas']  # Tail gas used as fuel (optional)
product_keywords = ['Hydrogen Product']

feedstock_streams = df[df['Description'].str.contains('|'.join(feedstock_keywords), case=False)]
fuel_streams = df[df['Description'].str.contains('|'.join(fuel_keywords), case=False)]
product_streams = df[df['Description'].str.contains('|'.join(product_keywords), case=False)]

# --- Calculate energy content ---
# Use component names that match the DataFrame columns (from HHV.keys())
feedstock_energy = feedstock_streams.apply(compute_stream_energy, components=components_for_calc, axis=1).sum()
fuel_energy = fuel_streams.apply(compute_stream_energy, components=components_for_calc, axis=1).sum()
total_input_energy = feedstock_energy + fuel_energy  # Total energy input
product_energy = product_streams.apply(compute_stream_energy, components=['H2'], axis=1).sum()

print("\n--- Feedstock Streams (LPG, Natural Gas) ---")
if len(feedstock_streams) > 0:
    for idx, row in feedstock_streams.iterrows():
        energy = compute_stream_energy(row, components_for_calc)
        print(f"{row['Description']}: {row['Molar_Flow_kmol_per_h']:.2f} kmol/h, Energy: {energy:.2f} MJ/h")
else:
    print("No feedstock streams found")

print("\n--- Fuel Streams (Tail Gas) ---")
if len(fuel_streams) > 0:
    for idx, row in fuel_streams.iterrows():
        energy = compute_stream_energy(row, components_for_calc)
        print(f"{row['Description']}: {row['Molar_Flow_kmol_per_h']:.2f} kmol/h, Energy: {energy:.2f} MJ/h")
else:
    print("No fuel streams found")

print("\n--- Product Streams ---")
print(product_streams[['Description', 'Molar_Flow_kmol_per_h', 'H2']])

# --- Calculate efficiency ---
print("\n--- Energy Balance ---")
print(f"Feedstock energy input: {feedstock_energy:.2f} MJ/h")
print(f"Fuel energy input: {fuel_energy:.2f} MJ/h")
print(f"Total energy input: {total_input_energy:.2f} MJ/h")
print(f"Hydrogen output energy: {product_energy:.2f} MJ/h")

if total_input_energy > 0:
    efficiency = (product_energy / total_input_energy) * 100
    print(f"\nThermal Efficiency: {efficiency:.2f} %")
    if efficiency > 100:
        print("WARNING: Efficiency > 100% indicates:")
        print("  - Missing energy inputs (steam, electricity, etc.)")
        print("  - Or incorrect feedstock/fuel identification")
else:
    print("\nERROR: Cannot calculate efficiency (no energy input found)")
