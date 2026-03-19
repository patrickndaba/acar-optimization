import streamlit as st
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="ACAR Framework Deployment", layout="wide")

st.title("ACAR Framework: Perishable Food Supply Chain Optimization")
st.markdown("---")

# --- LOAD DATA ---
@st.cache_resource
def load_assets():
    try:
        with open('route_options.pkl', 'rb') as f:
            route_options = pickle.load(f)
        return route_options
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None

route_options = load_assets()

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Shipment Parameters")

cities = sorted(list(set([u for u, v in route_options.keys()] + [v for u, v in route_options.keys()])))
source_city = st.sidebar.selectbox("Source City", cities, index=cities.index('Pune') if 'Pune' in cities else 0)
dest_city = st.sidebar.selectbox("Destination City", cities, index=cities.index('Delhi') if 'Delhi' in cities else 1)

food_product = st.sidebar.selectbox("Food Product", [
    'Milk', 'Fresh Vegetables', 'Meat', 'Fresh Fruits', 'Fish',
    'Frozen Peas', 'Ready-to-Eat Meals', 'Cheese', 'Paneer', 'Curd'
])

# --- SMART AI POLICY MAPPING ---
def get_automated_policy(product_name):
    high_perishables = ['Milk', 'Fresh Vegetables', 'Meat', 'Fresh Fruits', 'Fish']
    moderate_perishables = ['Frozen Peas', 'Ready-to-Eat Meals', 'Cheese', 'Paneer', 'Curd']
    if product_name in high_perishables: return (0.15, 0.75, 0.10) # Heavy Quality Focus
    elif product_name in moderate_perishables: return (0.40, 0.40, 0.20) # Balanced
    else: return (0.75, 0.10, 0.15) # Heavy Cost Focus

alpha, beta, gamma = get_automated_policy(food_product)
st.sidebar.markdown("---")
st.sidebar.success(f"**🤖 Smart AI Policy Active**\n\nOptimizing for **{food_product}**:\n- 💰 Cost: {alpha*100:.0f}%\n- ⏱️ Time: {beta*100:.0f}%\n- 🌱 Eco: {gamma*100:.0f}%")

# --- SMART GRAPH BUILDER ---
def build_optimized_graphs(options, a, b, g):
    G_std = nx.DiGraph()
    G_smart = nx.DiGraph()
    
    for (u, v), shipments in options.items():
        # 1. Standard: Find ABSOLUTE Cheapest shipment
        cheapest = min(shipments, key=lambda x: x['cost'])
        G_std.add_edge(u, v, weight=cheapest['cost'], cost=cheapest['cost'], time=cheapest['time'], carbon=cheapest['co2'])
        
        # 2. Smart AI: Find BEST shipment for this POLICY
        # Score = (a * cost) + (b * time * 500) + (g * carbon * 10)
        best_smart = min(shipments, key=lambda x: (a * x['cost']) + (b * x['time'] * 500) + (g * x['co2'] * 10))
        G_smart.add_edge(u, v, weight=(a * best_smart['cost']) + (b * best_smart['time'] * 500) + (g * best_smart['co2'] * 10), 
                         cost=best_smart['cost'], time=best_smart['time'], carbon=best_smart['co2'])
        
    return G_std, G_smart

# --- MAIN PAGE: OPTIMIZATION ---
if st.button("Optimize Route"):
    if route_options is not None:
        try:
            G_std, G_smart = build_optimized_graphs(route_options, alpha, beta, gamma)
            
            # 1. FIND ROUTES
            path_std = nx.shortest_path(G_std, source=source_city, target=dest_city, weight='weight')
            path_smart = nx.shortest_path(G_smart, source=source_city, target=dest_city, weight='weight')

            # 2. CALCULATE METRICS
            def get_path_metrics(path, G):
                t_cost, t_time, t_co2 = 0, 0, 0
                for i in range(len(path) - 1):
                    d = G[path[i]][path[i+1]]
                    t_cost += d['cost']; t_time += d['time']; t_co2 += d['carbon']
                return t_cost, t_time, t_co2

            c_std, t_std, e_std = get_path_metrics(path_std, G_std)
            c_smt, t_smt, e_smt = get_path_metrics(path_smart, G_smart)
            
            # 3. DISPLAY
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
                if path_std != path_smart:
                    nx.draw_networkx_edges(G_std, pos, edgelist=list(zip(path_std, path_std[1:])), edge_color='red', width=2, style='dashed', ax=ax)
                st.pyplot(fig)

            st.subheader("Physical Route Comparison: Standard vs. Smart AI")
            comparison_data = {
                "Metric": ["Path Taken", "Transportation Cost (INR)", "Total Transit Time (Hrs)", "CO2 Emissions (kg)"],
                "STANDARD (Cost-Min)": [" -> ".join(path_std), f"{c_std:,.2f}", f"{t_std:.2f} hrs", f"{e_std:.2f} kg"],
                "SMART AI (Multi-Objective)": [" -> ".join(path_smart), f"{c_smt:,.2f}", f"{t_smt:.2f} hrs", f"{e_smt:.2f} kg"]
            }
            st.table(pd.DataFrame(comparison_data))
            
            # Summary Metrics
            diff_cost = c_smt - c_std
            diff_time = t_smt - t_std
            col_a, col_b = st.columns(2)
            col_a.metric("Cost Difference", f"{diff_cost:,.2f}", delta=f"{(diff_cost/c_std)*100:.1f}%" if c_std != 0 else "N/A", delta_color="inverse")
            col_b.metric("Time Improvement", f"{diff_time:.2f} hrs", delta=f"{(diff_time/t_std)*100:.1f}%" if t_std != 0 else "N/A", delta_color="normal")

        except Exception as e:
            st.error(f"Route finding error: {e}")
    else:
        st.warning("Please ensure assets are loaded.")

st.sidebar.markdown("---")
st.sidebar.info("This app uses the ACAR Framework for multi-objective supply chain optimization.")
