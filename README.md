# ACAR Framework: Perishable Food Supply Chain Optimization

An AI-driven route optimization tool designed for perishable food supply chains, focusing on minimizing transportation costs, delivery time, and CO2 emissions.

## Project Overview
The ACAR (AI-Driven Route Optimization for Perishable Food) framework uses Random Forest regression and NetworkX graph algorithms to:
1.  **Forecast Transportation Costs:** Using shipment distance, fuel prices, and perishability factors.
2.  **Optimize Routes:** Utilizing a multi-objective shortest-path algorithm (Standard vs. Smart AI).
3.  **Minimize Emissions:** Factoring CO2 output into route selection for sustainable logistics.

## Key Features
-   **Multi-Objective Optimization:** Balances cost, quality preservation (shelf-life), and carbon footprint.
-   **Smart Weighting:** Dynamic weighting based on the type of food product (e.g., higher priority for highly perishable milk/meat).
-   **Comparison Table:** View the physical differences in cost, time, and CO2 between standard cost-minimization and AI-enhanced optimization.

## Installation & Setup

1.  **Clone the Repository**
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## Project Structure
-   `ACAR_Framework.ipynb`: Full research and analysis notebook.
-   `app.py`: Streamlit application for deployment.
-   `export_models.py`: Script for training and exporting model/graph assets.
-   `acar_rf_model.pkl`: Trained Random Forest model.
-   `preprocessor.pkl`: Data preprocessing pipeline.
-   `logistics_graph.gpickle`: Multi-objective transportation graph.
