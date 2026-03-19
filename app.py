import streamlit as st
import pandas as pd
import joblib
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="ACAR Framework Deployment", layout="wide")

st.title("ACAR Framework: Perishable Food Supply Chain Optimization")
st.markdown("---")

# --- LOAD DATA AND MODELS ---
@st.cache_resource
def load_assets():
    try:
        # Load the graph
        with open('logistics_graph.gpickle', 'rb') as f:
            G = pickle.load(f)
        
        # Load the Random Forest model and preprocessor
        rf = joblib.load('acar_rf_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        
        return G, rf, preprocessor
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

G, rf, preprocessor = load_assets()
# --- SIDEBAR: INPUTS ---
st.sidebar.header("Shipment Parameters")

source_city = st.sidebar.selectbox("Source City", ['Pune', 'Patna', 'Chandigarh', 'Ahmedabad', 'Kolkata', 'Chennai', 'Mumbai', 'Indore', 'Hyderabad', 'Delhi'])
dest_city = st.sidebar.selectbox("Destination City", ['Indore', 'Bhubaneswar', 'Raipur', 'Chennai', 'Delhi', 'Kolkata', 'Ranchi', 'Bhopal', 'Ahmedabad', 'Bengaluru'])
food_product = st.sidebar.selectbox("Food Product", ['Fresh Vegetables', 'Frozen Peas', 'Milk', 'Ready-to-Eat Meals', 'Cheese', 'Paneer', 'Fresh Fruits', 'Fish', 'Meat', 'Curd'])

st.sidebar.markdown("---")
st.sidebar.header("Optimization Policy")
alpha = st.sidebar.slider("Cost Weight (Alpha)", 0.0, 1.0, 0.4)
beta = st.sidebar.slider("Quality Weight (Beta)", 0.0, 1.0, 0.4)
gamma = st.sidebar.slider("Eco Weight (Gamma)", 0.0, 1.0, 0.2)

# Normalize weights
total_w = alpha + beta + gamma
if total_w > 0:
    alpha /= total_w
    beta /= total_w
    gamma /= total_w

# --- HELPER FUNCTIONS ---
def calculate_metrics(path, G):
    t_cost, t_time, t_co2 = 0, 0, 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        data = G[u][v]
        # Fallback to 0 if keys are missing
        t_cost += data.get('original_cost', data.get('weight', 0))
        t_time += data.get('time', 0)
        t_co2 += data.get('carbon', 0)
    return t_cost, t_time, t_co2

def calculate_custom_smart_weight(G, u, v, a, b, g):
    data = G[u][v]
    cost = data.get('original_cost', 0)
    time = data.get('time', 0)
    co2 = data.get('carbon', 0)
    # Simple composite score for live demo
    return (a * cost) + (b * time * 100) + (g * co2 * 10)

# --- MAIN PAGE: OPTIMIZATION ---
if st.button("Optimize Route"):
    if G is not None:
        try:
            # 1. FIND THE ROUTES
            # Standard Path: Minimum INR Cost
            path_standard = nx.shortest_path(G, source=source_city, target=dest_city, weight='original_cost')

            # Dynamic Smart Path: Using sidebar sliders
            # Create a temporary graph with the user's custom weights
            G_temp = G.copy()
            for u, v in G_temp.edges():
                G_temp[u][v]['custom_weight'] = calculate_custom_smart_weight(G_temp, u, v, alpha, beta, gamma)

            path_smart = nx.shortest_path(G_temp, source=source_city, target=dest_city, weight='custom_weight')

            # 2. CALCULATE METRICS
            cost_std, time_std, co2_std = calculate_metrics(path_standard, G)
            cost_smart, time_smart, co2_smart = calculate_metrics(path_smart, G)
            
            # 3. DISPLAY RESULTS
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimized Route")
                st.success(" -> ".join(path_smart))
            
            with col2:
                st.subheader("Visualization")
                fig, ax = plt.subplots(figsize=(8, 6))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_color='lightgrey', node_size=600, ax=ax)
                
                # Highlight Smart path
                smart_edges = list(zip(path_smart, path_smart[1:]))
                nx.draw_networkx_nodes(G, pos, nodelist=path_smart, node_color='green', ax=ax)
                nx.draw_networkx_edges(G, pos, edgelist=smart_edges, edge_color='green', width=4, alpha=0.6, ax=ax)
                
                # Highlight Standard path if different
                if path_standard != path_smart:
                    std_edges = list(zip(path_standard, path_standard[1:]))
                    nx.draw_networkx_edges(G, pos, edgelist=std_edges, edge_color='red', width=2, style='dashed', alpha=0.8, ax=ax)
                    st.caption("Green: Smart Route | Red Dashed: Standard Route")
                
                st.pyplot(fig)

            # 4. COMPARISON TABLE
            st.subheader("Physical Route Comparison: Standard vs. Smart AI")
            comparison_data = {
                "Metric": ["Path Taken", "Transportation Cost (INR)", "Total Transit Time (Hrs)", "CO2 Emissions (kg)"],
                "STANDARD (Cost-Min)": [
                    " -> ".join(path_standard), 
                    f"{cost_std:,.2f}", 
                    f"{time_std:.2f} hrs", 
                    f"{co2_std:.2f} kg"
                ],
                "SMART AI (Multi-Objective)": [
                    " -> ".join(path_smart), 
                    f"{cost_smart:,.2f}", 
                    f"{time_smart:.2f} hrs", 
                    f"{co2_smart:.2f} kg"
                ]
            }
            df_compare = pd.DataFrame(comparison_data)
            st.table(df_compare)
            
            # Summary Metrics with zero-division safety
            diff_cost = cost_smart - cost_std
            diff_co2 = co2_smart - co2_std
            
            col_a, col_b = st.columns(2)
            
            cost_delta = f"{(diff_cost/cost_std)*100:.1f}%" if cost_std != 0 else "N/A"
            co2_delta = f"{(diff_co2/co2_std)*100:.1f}%" if co2_std != 0 else "N/A"
            
            col_a.metric("Cost Difference", f"{diff_cost:,.2f}", delta=cost_delta, delta_color="inverse")
            col_b.metric("CO2 Change", f"{diff_co2:,.2f} kg", delta=co2_delta, delta_color="inverse")

        except nx.NetworkXNoPath:
            st.error("No path found between the selected cities!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please ensure assets are loaded.")

st.sidebar.markdown("---")
st.sidebar.info("This app uses the ACAR Framework for multi-objective supply chain optimization.")
