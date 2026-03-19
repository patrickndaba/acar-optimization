import pandas as pd
import joblib
import pickle
import networkx as nx
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# 1. Load data
df = pd.read_csv("Perishable_Food_Supply_Chain_Feature_Engineered.csv", index_col=0)

# 2. Define features
X = df.drop(columns=["Total_Transportation_Cost_INR"])
y = df["Total_Transportation_Cost_INR"]

# 3. Preprocessing
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 4. Train a minimal RF (fast)
X_processed = preprocessor.fit_transform(X)
rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_processed, y)

# 5. Build Graph
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row["Source_City"], row["Destination_City"], weight=row['Total_Transportation_Cost_INR'])

# 6. Save everything
joblib.dump(rf, 'acar_rf_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
with open('logistics_graph.gpickle', 'wb') as f:
    pickle.dump(G, f)

print("SUCCESS: Quick Export Complete!")
