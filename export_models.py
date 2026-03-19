import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle

# 1. LOAD DATA
df = pd.read_csv("Perishable_Food_Supply_Chain_Dataset.csv")

# 2. FEATURE ENGINEERING (CO2 Proxy)
emission_factors = {'Refrigerated Truck': 1.1, 'Non-Refrigerated Truck': 0.8, 'Mini Van': 0.5}
df['CO2_kg'] = df['Distance_km'] * df['Vehicle_Type'].map(emission_factors).fillna(0.9)

# 3. EXTRACT ALL SHIPMENT OPTIONS PER ROUTE
route_options = {}

for _, row in df.iterrows():
    u, v = row["Source_City"], row["Destination_City"]
    shipment = {
        'cost': row["Total_Transportation_Cost_INR"],
        'time': row["Transit_Time_hours"],
        'co2': row["CO2_kg"],
        'max_time': row["Max_Allowable_Time_hours"] # Added for Quality calculation
    }
    
    if (u, v) not in route_options:
        route_options[(u, v)] = []
    route_options[(u, v)].append(shipment)

# 4. SAVE
with open('route_options.pkl', 'wb') as f:
    pickle.dump(route_options, f)

print(f"SUCCESS: Exported {len(route_options)} routes with Max_Time for Quality analysis!")
