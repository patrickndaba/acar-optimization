import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import joblib
import pickle
import networkx as nx

# 1. LOAD DATA
df = pd.read_csv("Perishable_Food_Supply_Chain_Dataset.csv")

# 2. FEATURE ENGINEERING (As in notebook)
def get_automated_policy(product_name):
    high_perishables = ['Milk', 'Fresh Vegetables', 'Meat', 'Fresh Fruits', 'Fish']
    moderate_perishables = ['Frozen Peas', 'Ready-to-Eat Meals', 'Cheese', 'Paneer', 'Curd']
    if product_name in high_perishables: return (0.20, 0.70, 0.10)
    elif product_name in moderate_perishables: return (0.40, 0.40, 0.20)
    else: return (0.70, 0.10, 0.20)

def calculate_metrics(row):
    alpha, beta, gamma = get_automated_policy(row['Food_Product'])
    cost = row['Total_Transportation_Cost_INR']
    quality_penalty = (row['Transit_Time_hours'] / row['Max_Allowable_Time_hours']) * 10000
    emission_factors = {'Refrigerated Truck': 1.1, 'Non-Refrigerated Truck': 0.8, 'Mini Van': 0.5}
    carbon = row['Distance_km'] * emission_factors.get(row['Vehicle_Type'], 0.9)
    smart_weight = (alpha * cost) + (beta * quality_penalty) + (gamma * carbon)
    return pd.Series([smart_weight, carbon])

df[['Smart_Weight', 'CO2_kg']] = df.apply(calculate_metrics, axis=1)

# 3. BUILD DUAL GRAPH (One for Standard, one for Smart)
# Since nx.Graph only allows ONE edge between cities, we must pick the BEST for each objective.
G_standard = nx.Graph()
G_smart = nx.Graph()

for _, row in df.iterrows():
    u, v = row["Source_City"], row["Destination_City"]
    
    # For Standard: Keep the cheapest shipment for this route
    if not G_standard.has_edge(u, v) or row["Total_Transportation_Cost_INR"] < G_standard[u][v]['weight']:
        G_standard.add_edge(u, v, weight=row["Total_Transportation_Cost_INR"], 
                           cost=row["Total_Transportation_Cost_INR"], 
                           time=row["Transit_Time_hours"], 
                           carbon=row["CO2_kg"])
    
    # For Smart: Keep the best shipment according to Smart AI Weight
    if not G_smart.has_edge(u, v) or row["Smart_Weight"] < G_smart[u][v]['weight']:
        G_smart.add_edge(u, v, weight=row["Smart_Weight"], 
                        cost=row["Total_Transportation_Cost_INR"], 
                        time=row["Transit_Time_hours"], 
                        carbon=row["CO2_kg"])

# 4. SAVE ASSETS
with open('logistics_graph_std.gpickle', 'wb') as f:
    pickle.dump(G_standard, f)
with open('logistics_graph_smart.gpickle', 'wb') as f:
    pickle.dump(G_smart, f)

print("SUCCESS: Dual Graphs (Standard vs Smart) exported!")
