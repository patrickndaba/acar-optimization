import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import joblib
import pickle
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# 1. LOAD DATA
df = pd.read_csv("Perishable_Food_Supply_Chain_Dataset.csv")

# 2. FEATURE ENGINEERING
df['Cost_per_km'] = df["Total_Transportation_Cost_INR"] / df["Distance_km"]
df["Delay_Hours"] = np.maximum(df["Transit_Time_hours"] - df["Max_Allowable_Time_hours"], 0)
df["Delay_Ratio"] = df["Delay_Hours"] / df["Transit_Time_hours"]
df["Shelf_Life_Utilization"] = (df["Transit_Time_hours"] / df["Max_Allowable_Time_hours"])
df["Fuel_Cost_Index"] = (df["Distance_km"] * df["Fuel_Price_INR_per_litre"])
df["Perishability_Risk_Index"] = (df["Delay_Ratio"] + df["Shelf_Life_Utilization"]) / 2

# SMART WEIGHT CALCULATION
def calculate_smart_weight(row):
    high_perishables = ['Milk', 'Fresh Vegetables', 'Meat', 'Fresh Fruits']
    moderate_perishables = ['Frozen Peas', 'Ready-to-Eat Meals', 'Dairy Products']
    
    if row['Food_Product'] in high_perishables:
        alpha, beta, gamma = (0.20, 0.70, 0.10)
    elif row['Food_Product'] in moderate_perishables:
        alpha, beta, gamma = (0.40, 0.40, 0.20)
    else:
        alpha, beta, gamma = (0.70, 0.10, 0.20)
    
    cost = row['Total_Transportation_Cost_INR']
    quality_penalty = row['Shelf_Life_Utilization'] * 10000 
    emission_factors = {'Refrigerated Truck': 1.1, 'Non-Refrigerated Truck': 0.8, 'Mini Van': 0.5}
    carbon = row['Distance_km'] * emission_factors.get(row['Vehicle_Type'], 0.9)
    
    smart_weight = (alpha * cost) + (beta * quality_penalty) + (gamma * carbon)
    return pd.Series([smart_weight, carbon])

df[['Smart_Weight', 'CO2_kg']] = df.apply(calculate_smart_weight, axis=1)

# 3. TRAIN MODELS (Briefly for export)
X = df.drop(columns=["Total_Transportation_Cost_INR"])
y = df["Total_Transportation_Cost_INR"]
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), numerical_cols),
        ("cat", Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_processed, y_train)

# 4. EXPORT GRAPH
G = nx.Graph()
for _, row in df.iterrows():
    # Use the best (minimum smart weight) if multiple shipments exist for a pair
    u, v = row["Source_City"], row["Destination_City"]
    current_weight = row["Smart_Weight"]
    
    if G.has_edge(u, v):
        if current_weight < G[u][v]['weight']:
            G.add_edge(u, v, 
                       weight=current_weight, 
                       original_cost=row["Total_Transportation_Cost_INR"],
                       time=row["Transit_Time_hours"],
                       carbon=row["CO2_kg"])
    else:
        G.add_edge(u, v, 
                   weight=current_weight, 
                   original_cost=row["Total_Transportation_Cost_INR"],
                   time=row["Transit_Time_hours"],
                   carbon=row["CO2_kg"])

# 5. SAVE ASSETS
joblib.dump(rf, 'acar_rf_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
with open('logistics_graph.gpickle', 'wb') as f:
    pickle.dump(G, f)

print("SUCCESS: Cleaned Models and Full-Data Graph exported!")
