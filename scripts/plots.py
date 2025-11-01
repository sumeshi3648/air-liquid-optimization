import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = './data/LPG_EOR_Heat_Material_Balance.xlsx'
SHEET_NAME = 'All_Streams'
OUTPUT_DIR = 'data/plots'

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
df.dropna(how='all', inplace=True)

### PLOT 1: Mass Flow vs Temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Temperature_C', y='Mass_Flow_kg_per_h', hue='Description')
plt.title('Mass Flow vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Mass Flow (kg/h)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mass_flow_vs_temperature.png'))
plt.close()

### PLOT 2: Pressure Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Pressure_kgcm2g'].dropna(), bins=15, kde=True)
plt.title('Pressure Distribution')
plt.xlabel('Pressure (kg/cm²g)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pressure_distribution.png'))
plt.close()

### PLOT 3: Molar Flow vs Molecular Weight
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='MW_g_per_mol', y='Molar_Flow_kmol_per_h', hue='Description')
plt.title('Molar Flow vs Molecular Weight')
plt.xlabel('Molecular Weight (g/mol)')
plt.ylabel('Molar Flow (kmol/h)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'molar_flow_vs_mw.png'))
plt.close()

### PLOT 4: Average Mole Fractions
components = ['H2', 'CO', 'CO2', 'CH4', 'C2H6', 'C3H8']
df_melt = df.melt(id_vars=['Stream_No', 'Description'], value_vars=components,
                  var_name='Component', value_name='Mole_Fraction')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melt.dropna(), x='Component', y='Mole_Fraction', estimator=np.mean, errorbar=None)
plt.title('Average Mole Fractions of Major Components')
plt.ylabel('Average Mole Fraction')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'avg_mole_fractions.png'))
plt.close()

print(f"all plots saved to: {OUTPUT_DIR}/")
