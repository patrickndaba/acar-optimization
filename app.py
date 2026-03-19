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
        # Load the Dual Graphs
        with open('logistics_graph_std.gpickle', 'rb') as f:
            G_std = pickle.load(f)
        with open('logistics_graph_smart.gpickle', 'rb') as f:
            G_smart = pickle.load(f)
        return G_std, G_smart
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

G_std, G_smart = load_assets()

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Shipment Parameters")

source_city = st.sidebar.selectbox("Source City", ['Pune', 'Patna', 'Chandigarh', 'Ahmedabad', 'Kolkata', 'Chennai', 'Mumbai', 'Indore', 'Hyderabad', 'Delhi'])
dest_city = st.sidebar.selectbox("Destination City", ['Indore', 'Bhubaneswar', 'Raipur', 'Chennai', 'Delhi', 'Kolkata', 'Ranchi', 'Bhopal', 'Ahmedabad', 'Bengaluru'])
food_product = st.sidebar.selectbox("Food Product", ['Fresh Vegetables', 'Frozen Peas', 'Milk', 'Ready-to-Eat Meals', 'Cheese', 'Paneer', 'Fresh Fruits', 'Fish', 'Meat', 'Curd'])

# --- SMART AI POLICY MAPPING ---
def get_automated_policy(product_name):
    high_perishables = ['Milk', 'Fresh Vegetables', 'Meat', 'Fresh Fruits', 'Fish']
    moderate_perishables = ['Frozen Peas', 'Ready-to-Eat Meals', 'Cheese', 'Paneer', 'Curd']
    if product_name in high_perishables: return (0.20, 0.70, 0.10)
    elif product_name in moderate_perishables: return (0.40, 0.40, 0.20)
    else: return (0.70, 0.10, 0.20)

alpha, beta, gamma = get_automated_policy(food_product)
st.sidebar.markdown("---")
st.sidebar.success(f"**🤖 Smart AI Policy Active**\n\nOptimizing for **{food_product}**:\n- 💰 Cost: {alpha*100:.0f}%\n- ⏱️ Quality: {beta*100:.0f}%\n- 🌱 Eco: {gamma*100:.0f}%")

# --- HELPER FUNCTIONS ---
def get_path_metrics(path, G):
    t_cost, t_time, t_co2 = 0, 0, 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        data = G[u][v]
        t_cost += data.get('cost', 0)
        t_time += data.get('time', 0)
        t_co2 += data.get('carbon', 0)
    return t_cost, t_time, t_co2

# --- MAIN PAGE: OPTIMIZATION ---
if st.button("Optimize Route"):
    if G_std is not None and G_smart is not None:
        try:
            # 1. FIND THE ROUTES
            # Standard: Shortest path based on MIN COST shipment
            path_standard = nx.shortest_path(G_std, source=source_city, target=dest_city, weight='weight')
            # Smart AI: Shortest path based on MULTI-OBJECTIVE shipment
            path_smart = nx.shortest_path(G_smart, source=source_city, target=dest_city, weight='weight')

            # 2. CALCULATE METRICS
            cost_std, time_std, co2_std = get_path_metrics(path_standard, G_std)
            cost_smart, time_smart, co2_smart = get_path_metrics(path_smart, G_smart)
            
            # 3. DISPLAY RESULTS
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Optimized Route")
                st.success(" -> ".join(path_smart))
            
            with col2:
                st.subheader("Visualization")
                fig, ax = plt.subplots(figsize=(8, 6))
                pos = nx.spring_layout(G_smart, seed=42)
                nx.draw(G_smart, pos, with_labels=True, node_color='lightgrey', node_size=600, ax=ax)
                nx.draw_networkx_nodes(G_smart, pos, nodelist=path_smart, node_color='green', ax=ax)
                nx.draw_networkx_edges(G_smart, pos, edgelist=list(zip(path_smart, path_smart[1:])), edge_color='green', width=4, alpha=0.6, ax=ax)
                if path_standard != path_smart:
                    nx.draw_networkx_edges(G_std, pos, edgelist=list(zip(path_standard, path_standard[1:])), edge_color='red', width=2, style='dashed', ax=ax)
                st.pyplot(fig)

            # 4. COMPARISON TABLE
            st.subheader("Physical Route Comparison: Standard vs. Smart AI")
            comparison_data = {
                "Metric": ["Path Taken", "Transportation Cost (INR)", "Total Transit Time (Hrs)", "CO2 Emissions (kg)"],
                "STANDARD (Cost-Min)": [" -> ".join(path_standard), f"{cost_std:,.2f}", f"{time_std:.2f} hrs", f"{co2_std:.2f} kg"],
                "SMART AI (Multi-Objective)": [" -> ".join(path_smart), f"{cost_smart:,.2f}", f"{time_smart:.2f} hrs", f"{co2_smart:.2f} kg"]
            }
            st.table(pd.DataFrame(comparison_data))
            
            # Summary Metrics
            diff_cost = cost_smart - cost_std
            diff_co2 = co2_smart - co2_std
            col_a, col_b = st.columns(2)
            cost_delta = f"{(diff_cost/cost_std)*100:.1f}%" if cost_std != 0 else "N/A"
            co2_delta = f"{(diff_co2/co2_std)*100:.1f}%" if co2_std != 0 else "N/A"
            col_a.metric("Cost Difference", f"{diff_cost:,.2f}", delta=cost_delta, delta_color="inverse")
            col_b.metric("CO2 Change", f"{diff_co2:,.2f} kg", delta=co2_delta, delta_color="inverse")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please ensure assets are loaded.")

st.sidebar.markdown("---")
st.sidebar.info("This app uses the ACAR Framework for multi-objective supply chain optimization.")
