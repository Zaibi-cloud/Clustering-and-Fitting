import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Load dataset (replace with your dataset)
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
print("Loading dataset...")
df = pd.read_csv(url)
print("Dataset loaded successfully.")

# Data Cleaning and Selection
print("Cleaning data...")
df = df[['country', 'year', 'co2', 'gdp']].dropna()  # Keep only relevant columns and drop missing values
df = df[df['year'] > 1990]  # Filter data for years after 1990
print(f"Data cleaned. Remaining rows: {len(df)}")

# ---- RELATIONAL PLOT (Scatter Plot: CO2 vs GDP) ----
print("Generating scatter plot...")
sns.scatterplot(data=df, x='gdp', y='co2', hue='country', alpha=0.6, legend=False)  # Scatter plot of GDP vs CO2 emissions
plt.xscale('log')  # Log scale for GDP to better visualize variations
plt.xlabel("GDP (log scale)")
plt.ylabel("CO2 Emissions")
plt.title("CO2 Emissions vs GDP")
plt.show()
print("Scatter plot generated.")

# ---- CATEGORICAL PLOT (Bar Plot: CO2 Emissions by Country) ----
print("Generating bar plot...")
top_countries = df.groupby('country', as_index=False)['co2'].sum().nlargest(10, 'co2')  # Get top 10 countries by CO2 emissions
plt.bar(top_countries['country'], top_countries['co2'], color='skyblue')
plt.xlabel("Country")
plt.ylabel("Total CO2 Emissions")
plt.title("Top 10 Countries by CO2 Emissions")
plt.xticks(rotation=45, ha='right')  # Rotate country names for better readability
plt.show()
print("Bar plot generated.")

# ---- STATISTICAL PLOT (Correlation Heatmap) ----
print("Generating correlation heatmap...")
corr_matrix = df[['co2', 'gdp']].corr()  # Compute correlation matrix for CO2 and GDP
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")  # Display heatmap with correlation values
plt.title("Correlation Heatmap")
plt.show()
print("Correlation heatmap generated.")

# ---- STATISTICAL MOMENTS ----
print("Calculating statistical moments...")
co2_data = df['co2'].dropna()
print(f"Mean: {np.mean(co2_data):.2f}")  # Measure of central tendency
print(f"Variance: {np.var(co2_data):.2f}")  # Spread of data
print(f"Skewness: {skew(co2_data):.2f}")  # Asymmetry in data distribution
print(f"Kurtosis: {kurtosis(co2_data):.2f}")  # Measure of tail heaviness
print("Statistical moments calculated.")

# ---- CLUSTERING (K-Means on CO2 vs GDP) ----
print("Performing K-Means clustering...")
X = df[['gdp', 'co2']].dropna()  # Select features for clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)  # Apply K-Means clustering with 3 clusters
df.loc[X.index, 'Cluster'] = kmeans.labels_  # Assign cluster labels back to the dataframe
sns.scatterplot(data=df, x='gdp', y='co2', hue='Cluster', palette='viridis')  # Visualize clusters
plt.xscale('log')
plt.title("Clustering Countries by GDP and CO2 Emissions")
plt.show()
print("K-Means clustering completed.")

# ---- CURVE FITTING (Exponential Fit to CO2 Emissions over Time) ----
def exp_growth(x, a, b, c):
    return a * np.exp(b * (x - 1990)) + c  # Exponential function for curve fitting

print("Performing curve fitting...")
usa_data = df[df['country'] == 'United States'][['year', 'co2']].dropna()  # Filter data for the US
if len(usa_data) > 3:  # Ensure enough data points for curve fitting
    popt, _ = curve_fit(exp_growth, usa_data['year'], usa_data['co2'], maxfev=5000)  # Fit exponential curve
    plt.scatter(usa_data['year'], usa_data['co2'], label='Actual Data')  # Plot actual data points
    plt.plot(usa_data['year'], exp_growth(usa_data['year'], *popt), color='red', label='Fitted Curve')  # Plot fitted curve
    plt.xlabel("Year")
    plt.ylabel("CO2 Emissions")
    plt.title("Exponential Fit to CO2 Emissions in the US")
    plt.legend()
    plt.show()
    print("Curve fitting completed.")
else:
    print("Not enough data for curve fitting.")
