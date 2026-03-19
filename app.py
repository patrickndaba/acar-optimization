import streamlit as st
import pandas as pd
import joblib
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="ACAR Framework Deployment", layout="wide")

st.title("ðŸšš ACAR Framework: Perishable Food Supply Chain Optimization")
st.markdown("---")

# --- LOAD DATA AND MODELS ---
@st.cache_resource
def load_assets():
    try:
        # Load the graph (G_auto_smart which was exported as logistics_graph.gpickle)
        with open('logistics_graph.gpickle', 'rb') as f:
            G = pickle.load(f)
        
        # Load the Random Forest model and preprocessor (optional for now, as shortest path is in graph)
        rf = joblib.load('acar_rf_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        
        return G, rf, preprocessor
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

G, rf, preprocessor = load_assets()

# --- SIDEBAR: INPUTS ---
st.sidebar.header("ðŸ“¥ Shipment Parameters")

source_city = st.sidebar.selectbox("Source City", ['Pune', 'Patna', 'Chandigarh', 'Ahmedabad', 'Kolkata', 'Chennai', 'Mumbai', 'Indore', 'Hyderabad', 'Delhi'])
dest_city = st.sidebar.selectbox("Destination City", ['Indore', 'Bhubaneswar', 'Raipur', 'Chennai', 'Delhi', 'Kolkata', 'Ranchi', 'Bhopal', 'Ahmedabad', 'Bengaluru'])
food_product = st.sidebar.selectbox("Food Product", ['Fresh Vegetables', 'Frozen Peas', 'Milk', 'Ready-to-Eat Meals', 'Cheese', 'Paneer', 'Fresh Fruits', 'Fish', 'Meat', 'Curd'])

# --- HELPER FUNCTIONS ---
def calculate_metrics(path, G):
    t_cost, t_time, t_co2 = 0, 0, 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        data = G[u][v]
        t_cost += data['original_cost']
        t_time += data['time']
        t_co2 += data['carbon']
    return t_cost, t_time, t_co2

# --- MAIN PAGE: OPTIMIZATION ---
if st.button("ðŸš€ Optimize Route"):
    if G is not None:
        try:
            # 1. FIND THE ROUTES
            # Standard Path: Only cared about the lowest original INR cost
            path_standard = nx.shortest_path(G, source=source_city, target=dest_city, weight='original_cost')
            # Smart Path: Cares about Cost + Quality + CO2 (pre-calculated Smart_Weight as 'weight' in graph)
            path_smart = nx.shortest_path(G, source=source_city, target=dest_city, weight='weight')
            
            # 2. CALCULATE METRICS
            cost_std, time_std, co2_std = calculate_metrics(path_standard, G)
            cost_smart, time_smart, co2_smart = calculate_metrics(path_smart, G)
            
            # 3. DISPLAY RESULTS
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ðŸ“ Optimized Route")
                st.success(" âžœ ".join(path_smart))
            
            with col2:
                st.subheader("ðŸ“Š Visualization")
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
            st.subheader("ðŸ“Š Physical Route Comparison: Standard vs. Smart AI")
            comparison_data = {
                "Metric": ["Path Taken", "Transportation Cost (INR)", "Total Transit Time (Hrs)", "CO2 Emissions (kg)"],
                "STANDARD (Cost-Min)": [
                    " âžœ ".join(path_standard), 
                    f"â‚¹{cost_std:,.2f}", 
                    f"{time_std:.2f} hrs", 
                    f"{co2_std:.2f} kg"
                ],
                "SMART AI (Multi-Objective)": [
                    " âžœ ".join(path_smart), 
                    f"â‚¹{cost_smart:,.2f}", 
                    f"{time_smart:.2f} hrs", 
                    f"{co2_smart:.2f} kg"
                ]
            }
            df_compare = pd.DataFrame(comparison_data)
            st.table(df_compare)
            
            # Summary Metrics
            diff_cost = cost_smart - cost_std
            diff_co2 = co2_smart - co2_std
            
            col_a, col_b = st.columns(2)
            col_a.metric("Cost Difference", f"â‚¹{diff_cost:,.2f}", delta=f"{diff_cost/cost_std*100:.1f}%", delta_color="inverse")
            col_b.metric("CO2 Change", f"{diff_co2:,.2f} kg", delta=f"{diff_co2/co2_std*100:.1f}%", delta_color="inverse")

        except nx.NetworkXNoPath:
            st.error("No path found between the selected cities!")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please ensure models are exported and the graph is available.")

st.sidebar.markdown("---")
st.sidebar.info("This app uses the ACAR Framework for multi-objective supply chain optimization.")
