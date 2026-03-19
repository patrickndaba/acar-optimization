import streamlit as st
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ACAR Framework | Smart Logistics",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Global Font Change */
    html, body, [class*="css"], .stMarkdown, .main-header, .sub-header, h1, h2, h3, h4 {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* Professional Header */
    .main-header {
        color: #1e3a8a;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        color: #64748b;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1e3a8a !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }

    /* Section Cards */
    .css-1r6slb0, .e1tz724v0 {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #1e3a8a;
        color: white;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }

    /* Tables */
    .stTable {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

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

# --- HEADER SECTION ---
st.markdown('<h1 class="main-header">ACAR FRAMEWORK</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Driven Multi-Objective Optimization for Perishable Food Supply Chains</p>', unsafe_allow_html=True)

# --- SIDEBAR: CONTROL PANEL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=80) # Logistics Icon
    st.header("Shipment Parameters")
    
    if route_options:
        cities = sorted(list(set([u for u, v in route_options.keys()] + [v for u, v in route_options.keys()])))
        source_city = st.selectbox("Source City", cities, index=cities.index('Pune') if 'Pune' in cities else 0)
        dest_city = st.selectbox("Destination City", cities, index=cities.index('Delhi') if 'Delhi' in cities else 1)
        
        food_product = st.selectbox("Food Product", [
            'Milk', 'Fresh Vegetables', 'Meat', 'Fresh Fruits', 'Fish',
            'Frozen Peas', 'Ready-to-Eat Meals', 'Cheese', 'Paneer', 'Curd'
        ])

        # Policy Mapping
        def get_automated_policy(product_name):
            high_perishables = ['Milk', 'Fresh Vegetables', 'Meat', 'Fresh Fruits', 'Fish']
            moderate_perishables = ['Frozen Peas', 'Ready-to-Eat Meals', 'Cheese', 'Paneer', 'Curd']
            if product_name in high_perishables: return (0.15, 0.75, 0.10)
            elif product_name in moderate_perishables: return (0.40, 0.40, 0.20)
            else: return (0.75, 0.10, 0.15)

        alpha, beta, gamma = get_automated_policy(food_product)
        
        st.markdown("---")
        st.markdown("### 🤖 Smart AI Policy")
        st.info(f"**Target:** {food_product}\n\n"
                f"- 💰 **Cost:** {alpha*100:.0f}%\n"
                f"- ⏱️ **Quality:** {beta*100:.0f}%\n"
                f"- 🌱 **Sustainability:** {gamma*100:.0f}%")
        
        optimize_btn = st.button("RUN OPTIMIZATION")

# --- SMART GRAPH BUILDER ---
def build_optimized_graphs(options, a, b, g):
    G_std = nx.DiGraph()
    G_smart = nx.DiGraph()
    for (u, v), shipments in options.items():
        # Standard
        cheapest = min(shipments, key=lambda x: x['cost'])
        G_std.add_edge(u, v, weight=cheapest['cost'], cost=cheapest['cost'], time=cheapest['time'], carbon=cheapest['co2'], max_time=cheapest['max_time'])
        # Smart
        best_smart = min(shipments, key=lambda x: (a * x['cost']) + (b * (x['time']/x['max_time']) * 10000) + (g * x['co2'] * 10))
        G_smart.add_edge(u, v, weight=(a * best_smart['cost']) + (b * (best_smart['time']/best_smart['max_time']) * 10000) + (g * best_smart['co2'] * 10), 
                         cost=best_smart['cost'], time=best_smart['time'], carbon=best_smart['co2'], max_time=best_smart['max_time'])
    return G_std, G_smart

# --- MAIN PAGE: DASHBOARD ---
if route_options and ('optimize_btn' in locals() and optimize_btn):
    try:
        G_std, G_smart = build_optimized_graphs(route_options, alpha, beta, gamma)
        path_std = nx.shortest_path(G_std, source=source_city, target=dest_city, weight='weight')
        path_smart = nx.shortest_path(G_smart, source=source_city, target=dest_city, weight='weight')

        def get_path_metrics(path, G):
            t_cost, t_time, t_co2, t_max_time = 0, 0, 0, 0
            for i in range(len(path) - 1):
                d = G[path[i]][path[i+1]]
                t_cost += d['cost']; t_time += d['time']; t_co2 += d['carbon']; t_max_time += d['max_time']
            risk = (t_time / t_max_time * 100) if t_max_time > 0 else 0
            quality = max(100 - risk, 0)
            return t_cost, t_time, t_co2, quality, risk

        c_std, t_std, e_std, q_std, r_std = get_path_metrics(path_std, G_std)
        c_smt, t_smt, e_smt, q_smt, r_smt = get_path_metrics(path_smart, G_smart)

        # 1. KEY PERFORMANCE INDICATORS
        st.markdown("### 📊 Performance Summary")
        kpi1, kpi2, kpi3 = st.columns(3)
        
        diff_cost = c_smt - c_std
        diff_qual = q_smt - q_std
        diff_co2 = e_smt - e_std
        
        kpi1.metric("Transportation Cost", f"₹{c_smt:,.0f}", delta=f"{(diff_cost/c_std)*100:+.1f}% vs Std", delta_color="inverse")
        kpi2.metric("Quality Preservation", f"{q_smt:.1f}%", delta=f"{diff_qual:+.1f}% vs Std", delta_color="normal")
        kpi3.metric("Carbon Footprint", f"{e_smt:,.1f} kg", delta=f"{(diff_co2/e_std)*100:+.1f}% vs Std", delta_color="inverse")

        # 2. VISUALIZATION & ROUTE
        col_map, col_table = st.columns([1, 1])
        
        with col_map:
            st.markdown("#### 🗺️ Interactive Network Visualization")
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G_smart, seed=42)
            nx.draw(G_smart, pos, with_labels=True, node_color='#f1f5f9', edge_color='#e2e8f0', 
                   node_size=1000, font_size=10, font_weight='bold', ax=ax)
            
            # Draw standard path shadow
            std_edges = list(zip(path_std, path_std[1:]))
            nx.draw_networkx_edges(G_std, pos, edgelist=std_edges, edge_color='#ef4444', width=8, alpha=0.2, ax=ax)
            
            # Draw smart path
            smart_edges = list(zip(path_smart, path_smart[1:]))
            nx.draw_networkx_edges(G_smart, pos, edgelist=smart_edges, edge_color='#10b981', width=4, alpha=0.8, ax=ax)
            nx.draw_networkx_nodes(G_smart, pos, nodelist=path_smart, node_color='#10b981', node_size=1200, ax=ax)
            
            ax.set_title(f"Optimized Route for {food_product}", color="#1e3a8a", fontsize=14, fontweight='bold')
            st.pyplot(fig)

        with col_table:
            st.markdown("#### 📋 Detailed Comparison Matrix")
            comparison_data = {
                "Metric": ["Path Taken", "Cost (INR)", "Quality (%)", "Risk (%)", "CO2 (kg)"],
                "Standard AI": [" → ".join(path_std), f"₹{c_std:,.2f}", f"{q_std:.1f}%", f"{r_std:.1f}%", f"{e_std:.1f}"],
                "Smart AI": [" → ".join(path_smart), f"₹{c_smt:,.2f}", f"{q_smt:.1f}%", f"{r_smt:.1f}%", f"{e_smt:.1f}"]
            }
            st.table(pd.DataFrame(comparison_data))
            
            # Insight message
            if path_std == path_smart:
                st.info("💡 **AI Insight:** The optimal physical path is identical, but the Smart AI selected a more efficient vehicle/speed combination to maximize freshness.")
            else:
                st.success(f"🚀 **AI Insight:** Smart AI discovered a more efficient path through {path_smart[1:-1]} to reduce perishability risk by {abs(diff_qual):.1f}%.")

    except Exception as e:
        st.error(f"⚠️ Optimization Error: {e}")

else:
    # Landing State
    st.markdown("---")
    st.info("👈 **Welcome to the ACAR Command Center.** Please select your shipment parameters in the sidebar and click **RUN OPTIMIZATION** to begin.")
    
    # Feature Cards
    f1, f2, f3 = st.columns(3)
    f1.markdown("### 🚛 Multi-Objective\nBalancing cost, time, and emissions in one powerful algorithm.")
    f2.markdown("### 🥬 Perishability-Aware\nSmart sensors and data models ensure food quality is preserved.")
    f3.markdown("### 🌱 Sustainable\nOptimized for the lowest carbon footprint without sacrificing speed.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8;'>ACAR Framework Deployment © 2026 | Perishable Food Supply Chain Optimization</p>", unsafe_allow_html=True)
