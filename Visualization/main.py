import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Load the dataset
def load_data(file_path):
    """Loads the dataset into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

# Explore the dataset
def explore_data(df):
    """Performs basic exploratory data analysis."""
    if df is not None:
        print("\nDataset Overview:\n")
        print(df.head())
        print("\nSummary Statistics:\n")
        print(df.describe())
        print("\nMissing Values:\n")
        print(df.isnull().sum())
    else:
        print("Data not loaded. Cannot explore.")

# Visualize missing values
def visualize_missing_values(df, output_path):
    """Creates a heatmap for missing values and saves it as a PNG file."""
    if df is not None:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.savefig(f"{output_path}/missing_values_heatmap.png")
        plt.close()

# Distribution of numerical features
def plot_numerical_distributions(df, numerical_columns, output_path):
    """Plots histograms for numerical columns and saves them as PNG files."""
    if df is not None:
        for col in numerical_columns:
            if col in df.columns:
                plt.figure(figsize=(8, 4))
                sns.histplot(df[col], kde=True, color='blue')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.savefig(f"{output_path}/{col}_distribution.png")
                plt.close()

                # Plotly interactive histogram
                fig = px.histogram(df, x=col, nbins=30, title=f"Interactive Distribution of {col}")
                fig.write_html(f"{output_path}/{col}_distribution_interactive.html")
            else:
                print(f"Column '{col}' not found in the dataset.")

# Categorical data distribution
def plot_categorical_distributions(df, categorical_columns, output_path):
    """Plots bar charts for categorical columns and saves them as PNG and interactive HTML files."""
    if df is not None:
        for col in categorical_columns:
            if col in df.columns:
                plt.figure(figsize=(8, 4))
                sns.countplot(data=df, x=col, palette='viridis')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.savefig(f"{output_path}/{col}_distribution.png")
                plt.close()

                # Plotly interactive bar chart
                fig = px.bar(df, x=col, title=f"Interactive Distribution of {col}")
                fig.write_html(f"{output_path}/{col}_distribution_interactive.html")
            else:
                print(f"Column '{col}' not found in the dataset.")

# Correlation heatmap
def plot_correlation_heatmap(df, numerical_columns, output_path):
    """Plots a heatmap showing correlations between numerical columns and saves it as a PNG file."""
    if df is not None:
        available_cols = [col for col in numerical_columns if col in df.columns]
        if available_cols:
            plt.figure(figsize=(10, 8))
            corr = df[available_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Heatmap')
            plt.savefig(f"{output_path}/correlation_heatmap.png")
            plt.close()

            # Plotly interactive heatmap
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='coolwarm', title="Interactive Correlation Heatmap")
            fig.write_html(f"{output_path}/correlation_heatmap_interactive.html")
        else:
            print("No valid numerical columns for correlation heatmap.")

# Scatter plots for numerical relationships
def plot_scatter_relationships(df, numerical_pairs, output_path):
    """Creates scatter plots for given numerical column pairs and saves them as PNG and interactive HTML files."""
    if df is not None:
        for pair in numerical_pairs:
            if pair[0] in df.columns and pair[1] in df.columns:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df, x=pair[0], y=pair[1], hue=None, palette='viridis')
                plt.title(f'{pair[0]} vs {pair[1]}')
                plt.xlabel(pair[0])
                plt.ylabel(pair[1])
                plt.savefig(f"{output_path}/{pair[0]}_vs_{pair[1]}.png")
                plt.close()

                # Plotly interactive scatter plot
                fig = px.scatter(df, x=pair[0], y=pair[1], title=f"Interactive Scatter Plot: {pair[0]} vs {pair[1]}")
                fig.write_html(f"{output_path}/{pair[0]}_vs_{pair[1]}_interactive.html")
            else:
                print(f"Columns '{pair[0]}' or '{pair[1]}' not found in the dataset.")

# Pie chart for category proportions
def plot_pie_chart(df, column, output_path):
    """Creates a pie chart for a categorical column and saves it as PNG and interactive HTML files."""
    if df is not None:
        if column in df.columns:
            pie_data = df[column].value_counts()
            plt.figure(figsize=(8, 8))
            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
            plt.title(f'Proportion of {column}')
            plt.savefig(f"{output_path}/{column}_proportion.png")
            plt.close()

            # Plotly interactive pie chart
            fig = px.pie(df, names=column, title=f"Interactive Proportion of {column}")
            fig.write_html(f"{output_path}/{column}_proportion_interactive.html")
        else:
            print(f"Column '{column}' not found in the dataset.")

# Main function to execute the visualization pipeline
def main(file_path, output_path):
    """Executes the visualization pipeline for the provided dataset."""
    df = load_data(file_path)

    if df is not None:
        # Initial Exploration
        explore_data(df)

        # Visualize missing values
        visualize_missing_values(df, output_path)

        # Specify numerical and categorical columns
        numerical_columns = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']  # Adjusted for heart disease dataset
        categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

        # Plot numerical distributions
        plot_numerical_distributions(df, numerical_columns, output_path)

        # Plot categorical distributions
        plot_categorical_distributions(df, categorical_columns, output_path)

        # Correlation heatmap
        plot_correlation_heatmap(df, numerical_columns, output_path)

        # Scatter plot relationships
        numerical_pairs = [('age', 'thalch'), ('chol', 'trestbps')]  # Example pairs for heart disease dataset
        plot_scatter_relationships(df, numerical_pairs, output_path)

        # Pie chart for categorical proportions
        plot_pie_chart(df, 'sex', output_path)  # Adjust column name as needed

        print("Analysis completed. All plots saved to the output directory, including interactive visualizations.")
    else:
        print("Analysis failed due to missing or incorrect data.")

# Example usage
file_path = 'D:/heart_disease_prediction/data/transformed_heart_disease.csv'  # Replace with your heart disease dataset file path
output_path = 'Visualization'  # Replace with your desired output directory

if not os.path.exists(output_path):
    os.makedirs(output_path)

main(file_path, output_path)
