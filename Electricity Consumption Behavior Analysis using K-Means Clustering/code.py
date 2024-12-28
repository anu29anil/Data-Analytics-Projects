import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to handle clustering and analysis
def analyze_data(csv_path):
    try:
        # Load the data
        data = pd.read_csv(csv_path)
        print("Data loaded. Shape:", data.shape)

        # Prepare features for clustering
        features = ['cons_12m', 'forecast_cons_12m', 'forecast_price_energy_peak', 'forecast_price_energy_off_peak']
        X = data[features].copy()  # Create a copy of the DataFrame slice

        # Check for infinite values in the entire DataFrame and report them
        if (X == np.inf).any().any() or (X == -np.inf).any().any():
            print("Infinite values found in the dataset. Replacing with NaN...")
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check for missing values
        if X.isna().any().any():
            print("Missing values found in feature set, filling with column means...")
            X.fillna(X.mean(), inplace=True)

        # Cap extreme outliers at the 99th percentile
        for col in features:
            upper_limit = X[col].quantile(0.99)
            X[col] = np.where(X[col] > upper_limit, upper_limit, X[col])  # Cap outliers

        # Check for any infinite or NaN values after preprocessing
        if X.isna().any().any() or (X == np.inf).any().any():
            raise ValueError("Data still contains infinite or NaN values after preprocessing.")

        # Convert all values to float64 (ensure numerical consistency)
        X = X.astype(np.float64)

        # Check for extremely large values that might cause issues
        if (X.abs() > np.finfo(np.float64).max).any().any():
            raise ValueError("Data contains values too large for dtype('float64').")

        # Normalize the features using RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        data['cluster'] = kmeans.fit_predict(X_scaled)

        # Calculate additional metrics
        data['coefficient_of_variation'] = data.groupby('cluster')['cons_12m'].transform(lambda x: x.std() / x.mean())
        data['electricity_fee_recovery_rate'] = data['forecast_cons_12m'] / data['cons_12m']
        data['growth_rate'] = (data['forecast_cons_12m'] - data['cons_12m']) / data['cons_12m']
        data['load_rate'] = data['cons_12m'] / data['cons_12m'].max()
        data['avg_electricity_price'] = (data['forecast_price_energy_peak'] + data['forecast_price_energy_off_peak']) / 2
        data['electricity_purchase'] = data['cons_12m'] * data['avg_electricity_price']

        # Rename clusters based on your custom names
        cluster_names = {
            0: 'High-value customers',
            1: 'High-potential customers',
            2: 'Ordinary customers',
            3: 'Low-value customers'
        }
        data['cluster'] = data['cluster'].map(cluster_names)

        # Calculate mean values for each cluster with the updated metrics
        cluster_means = data.groupby('cluster').agg({
            'cons_12m': 'mean',
            'avg_electricity_price': 'mean',
            'growth_rate': 'mean',
            'coefficient_of_variation': 'mean',
            'load_rate': 'mean'
        }).reset_index()

        # Define features to plot with full names
        features_to_plot = {
            'cons_12m': '12-Month Consumption',
            'avg_electricity_price': 'Average Electricity Price',
            'growth_rate': 'Growth Rate',
            'coefficient_of_variation': 'Coefficient of Variation',
            'load_rate': 'Load Rate'
        }

        # Normalize features for visualization
        cluster_means[list(features_to_plot.keys())] = scaler.fit_transform(cluster_means[list(features_to_plot.keys())])

        # Create a new window for the plot
        plot_window = tk.Toplevel()
        plot_window.title("Customer Group Analysis")
        plot_window.geometry("1000x700")  # Larger window size to accommodate the chart
        plot_window.configure(bg="#ffe4e1")  # Set window color to light pink

        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 10))  # Adjusted figure size for better visibility

        # Define colors for each feature
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF9966']

        # Plot horizontal bars for each customer group
        bar_height = 0.15
        y_positions = np.arange(len(cluster_means)) * (len(features_to_plot) + 1) * bar_height

        for i, (col, full_name) in enumerate(features_to_plot.items()):
            ax.barh(y_positions + i * bar_height, cluster_means[col], height=bar_height, color=colors[i], alpha=0.7, label=full_name)

            # Add value labels on the bars
            for j, value in enumerate(cluster_means[col]):
                ax.text(value, y_positions[j] + i * bar_height, f'{value:.2f}', va='center', ha='left', fontsize=10)

        # Customize the plot
        ax.set_yticks(y_positions + (len(features_to_plot) - 1) * bar_height / 2)
        ax.set_yticklabels(cluster_means['cluster'], fontsize=12)  # Adjust y-tick label font size
        ax.invert_yaxis()  # Invert y-axis to match the image order
        ax.set_xlabel('Normalized Value', fontsize=12)
        ax.set_title('Analysis of Clustering Results of Electricity Customer Groups', fontsize=16)

        # Adjust the legend and make sure it's fully visible
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Adjust the plot layout to prevent label overlap
        fig.tight_layout(pad=4.0)  # Added extra padding to prevent clipping of labels

        # Create a canvas to embed the matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to analyze data: {e}")

# Function to open file dialog and get the CSV file path
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

# Main Tkinter window
root = tk.Tk()
root.title("Electricity Customer Group Analyzer")
root.geometry("500x260")  # Fixed size for the main window
root.configure(bg="#ffe4e1")  # Set background color to light pink

# Add a title label
title_label = tk.Label(root, text="Electricity Consumption Behaviour Analyzer", font=("Arial", 14, "bold"), bg="#ffe4e1")
title_label.pack(pady=20)

# Create and place labels and buttons
tk.Label(root, text="Enter the CSV file path:", font=("Arial", 12), bg="#ffe4e1").pack(pady=10)
file_entry = tk.Entry(root, width=50, font=("Arial", 10))
file_entry.pack(pady=5)

browse_button = tk.Button(root, text="Browse", command=browse_file, font=("Arial", 10, "bold"), bg="white", fg="black", relief="ridge")
browse_button.pack(pady=5)

analyze_button = tk.Button(root, text="Analyse", command=lambda: analyze_data(file_entry.get()), font=("Arial", 10, "bold"), bg="white", fg="black", relief="ridge")
analyze_button.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()
