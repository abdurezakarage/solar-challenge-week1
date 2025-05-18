import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
import os
if __name__ == "__main__":
   
    df_soeraleone = pd.read_csv("data/sierraleone-bumbuna.csv")
    df_soeraleone['Timestamp'] = pd.to_datetime(df_soeraleone['Timestamp'])
    #summary stats
    df_soeraleone.describe()
     #Missing values analysis
    soeraleone_missing_values = df_soeraleone.isnull().sum()
    soeraleone_missing_values = soeraleone_missing_values[soeraleone_missing_values > 0]
    #print(soeraleone_missing_values)
    # Comment columns empty so we drop them
    df_soeraleone = df_soeraleone.drop(columns=['Comments'])
    # Drop the 'comments' column if it exists
    if 'Comments' in df_soeraleone.columns:
        df_soeraleone = df_soeraleone.drop(columns=['Comments'])
        "Dropped 'Comments' column"
    else:
        "No 'Comments' column found in dataset"
    # visualize distribution of data
key_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust','Precipitation','TModA','TModB']

plt.figure(figsize=(14, 12))
for i, col in enumerate(key_cols):
    plt.subplot(4, 3, i+1)
    sns.boxplot(data=df_soeraleone, x=col)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
#plt.show()
# ------------------------------
# Compute Z-scores to Detect Outliers
z_scores = df_soeraleone[key_cols].apply(zscore)
# Flag rows where |Z| > 3
soeraleone_outlier_mask = z_scores.abs() > 3
soeraleone_outlier_counts = soeraleone_outlier_mask.sum()


# total number of rows flagged as outliers in any key column
soeraleone_outlier_mask.sum()
#create graph for total outlier in the data
plt.figure(figsize=(10, 6))
sns.barplot(x=soeraleone_outlier_counts.index, y=soeraleone_outlier_counts.values)
plt.title('Total Outliers in Each Column')
plt.xlabel('Columns')
plt.ylabel('Number of Outliers')
plt.tight_layout()
#plt.show()

# Find all rows that have any outliers in key columns
rows_with_outliers = soeraleone_outlier_mask.any(axis=1)

# Drop the outlier rows
df_soeraleone_clean = df_soeraleone[~rows_with_outliers]
#print(f"Total rows after dropping outliers: {len(df_soeraleone_clean)}")

# Verify the data is clean by checking for outliers again
z_scores_clean = df_soeraleone_clean[key_cols].apply(zscore)
soeraleone_clean_outlier_mask = z_scores_clean.abs() > 3
soeraleone_clean_outlier_count = soeraleone_clean_outlier_mask.any(axis=1).sum()
#print(f"Remaining outliers after cleaning: {soeraleone_clean_outlier_count}")

# Implement iterative outlier removal until convergence
#print("\nPerforming iterative outlier removal...")
iteration = 1
max_iterations = 5
previous_count = len(df_soeraleone_clean)

while soeraleone_clean_outlier_count > 0 and iteration <= max_iterations:
    # Remove outliers
    df_soeraleone_clean = df_soeraleone_clean[~soeraleone_clean_outlier_mask.any(axis=1)]
    
    # Recalculate Z-scores and outliers
    z_scores_clean = df_soeraleone_clean[key_cols].apply(zscore)
    soeraleone_clean_outlier_mask = z_scores_clean.abs() > 3
    soeraleone_clean_outlier_count = soeraleone_clean_outlier_mask.any(axis=1).sum()
    
    # Print results
    current_count = len(df_soeraleone_clean)
    removed = previous_count - current_count
    # print(f"Iteration {iteration}: Removed {removed} rows, {soeraleone_clean_outlier_count} outliers remaining")
    
    # Update for next iteration
    previous_count = current_count
    iteration += 1

#print(f"\nFinal clean dataset: {len(df_soeraleone_clean)} rows")
#print(f"Total rows removed: {len(df_soeraleone) - len(df_soeraleone_clean)}")

# Save the cleaned dataset
clean_file_path = "data/sierraleone-bumbuna-clean.csv"
df_soeraleone_clean.to_csv(clean_file_path, index=False)
#print(f"Cleaned data saved to {clean_file_path}")

# # Compare data statistics before and after cleaning
#print("\nStatistics before cleaning:")
#print(df_soeraleone[key_cols].describe().T[['mean', 'std', 'min', 'max']])

#print("\nStatistics after cleaning:")
#print(df_soeraleone_clean[key_cols].describe().T[['mean', 'std', 'min', 'max']])

# Create a better visualization to compare before and after cleaning
plt.figure(figsize=(18, 15))

# For each statistic, create a subplot to compare before and after
stats_to_compare = ['mean', 'std', 'min', 'max']
for i, stat in enumerate(stats_to_compare):
    plt.subplot(2, 2, i+1)
    
    # Get the values before and after cleaning
    before_values = df_soeraleone[key_cols].describe().T[stat]
    after_values = df_soeraleone_clean[key_cols].describe().T[stat]
    
    # Create a DataFrame for easy plotting
    compare_df = pd.DataFrame({
        'Before Cleaning': before_values,
        'After Cleaning': after_values
    })
    
    # Plot the comparison
    compare_df.plot(kind='bar', ax=plt.gca())
    plt.title(f'Comparison of {stat.capitalize()} Values Before and After Cleaning')
    plt.ylabel(f'{stat.capitalize()} Value')
    plt.xticks(rotation=45)
    plt.legend()

plt.tight_layout()
#plt.show()

# Also create a comparison of data distributions using boxplots for a few key columns
plt.figure(figsize=(16, 12))
for i, col in enumerate(key_cols[:6]):  # Just show first 6 columns to avoid crowding
    plt.subplot(2, 3, i+1)
    
    # Plot both original and cleaned data on same axis
    sns.boxplot(data=[df_soeraleone[col], df_soeraleone_clean[col]])
    plt.title(f'{col} Distribution Comparison')
    plt.xticks([0, 1], ['Original', 'Cleaned'])

plt.tight_layout()
#plt.show()

# Time Series Analysis for cleaned data

df_soeraleone_clean['Timestamp'] = pd.to_datetime(df_soeraleone_clean['Timestamp'])
#how many columns are in the dataset
# Extract date components for grouping
df_soeraleone_clean['Date'] = df_soeraleone_clean['Timestamp'].dt.date
df_soeraleone_clean['Month'] = df_soeraleone_clean['Timestamp'].dt.month
df_soeraleone_clean['Day'] = df_soeraleone_clean['Timestamp'].dt.day
df_soeraleone_clean['Hour'] = df_soeraleone_clean['Timestamp'].dt.hour

# 1. Daily patterns - Mean values by hour of day
daily_patterns = df_soeraleone_clean.groupby('Hour')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
daily_patterns[['GHI', 'DNI', 'DHI']].plot(kind='line', ax=plt.gca())
plt.title('Mean Solar Irradiance by Hour of Day')
plt.ylabel('Irradiance (W/m²)')
plt.grid(True)

plt.subplot(2, 1, 2)
daily_patterns['Tamb'].plot(kind='line', color='red', ax=plt.gca())
plt.title('Mean Ambient Temperature by Hour of Day')
plt.ylabel('Temperature (°C)')
plt.xlabel('Hour of Day')
plt.grid(True)
plt.tight_layout()
#plt.show()

# 2. Monthly patterns - Mean values by month
monthly_patterns = df_soeraleone_clean.groupby('Month')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
monthly_patterns[['GHI', 'DNI', 'DHI']].plot(kind='bar', ax=plt.gca())
plt.title('Mean Solar Irradiance by Month')
plt.ylabel('Irradiance (W/m²)')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, axis='y')

plt.subplot(2, 1, 2)
monthly_patterns['Tamb'].plot(kind='bar', color='red', ax=plt.gca())
plt.title('Mean Ambient Temperature by Month')
plt.ylabel('Temperature (°C)')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, axis='y')
plt.tight_layout()
#plt.show()
# Cleaning impact

# Create a DataFrame to compare pre and post cleaning values
print("\nAnalyzing cleaning impact on module values...")

# First, create a combined dataset for comparison
# Create a copy of the original data with a "Before" flag
df_before = df_soeraleone.copy()
df_before['Cleaning'] = 'Before'

# Create a copy of the cleaned data with an "After" flag
df_after = df_soeraleone_clean.copy()
df_after['Cleaning'] = 'After'

# Combine both datasets
combined_df = pd.concat([df_before, df_after])

# Calculate module statistics before and after cleaning
module_stats = combined_df.groupby('Cleaning')[['ModA', 'ModB']].agg(['mean', 'std', 'min', 'max'])
print("\nModule statistics before and after cleaning:")
print(module_stats)

# Create visualization for ModA and ModB before and after cleaning
plt.figure(figsize=(14, 10))

# Plot ModA comparison - change to bar graph
plt.subplot(2, 2, 1)
module_means = combined_df.groupby('Cleaning')['ModA'].mean()
module_means.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'lightgreen'])
plt.title('Impact of Cleaning on ModA (Mean)')
plt.ylabel('ModA Mean (W/m²)')
plt.grid(True, axis='y')
plt.xticks(rotation=0)

# Plot ModB comparison - change to bar graph
plt.subplot(2, 2, 2)
module_means = combined_df.groupby('Cleaning')['ModB'].mean()
module_means.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'lightgreen'])
plt.title('Impact of Cleaning on ModB (Mean)')
plt.ylabel('ModB Mean (W/m²)')
plt.grid(True, axis='y')
plt.xticks(rotation=0)

# Plot bar charts for ModA distribution metrics
plt.subplot(2, 2, 3)
moda_stats = pd.DataFrame({
    'Before': [df_before['ModA'].mean(), df_before['ModA'].std(), df_before['ModA'].median()],
    'After': [df_after['ModA'].mean(), df_after['ModA'].std(), df_after['ModA'].median()]
}, index=['Mean', 'Std Dev', 'Median'])
moda_stats.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'lightgreen'])
plt.title('ModA Distribution Metrics')
plt.ylabel('Value (W/m²)')
plt.grid(True, axis='y')
plt.legend()

# Plot bar charts for ModB distribution metrics
plt.subplot(2, 2, 4)
modb_stats = pd.DataFrame({
    'Before': [df_before['ModB'].mean(), df_before['ModB'].std(), df_before['ModB'].median()],
    'After': [df_after['ModB'].mean(), df_after['ModB'].std(), df_after['ModB'].median()]
}, index=['Mean', 'Std Dev', 'Median'])
modb_stats.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'lightgreen'])
plt.title('ModB Distribution Metrics')
plt.ylabel('Value (W/m²)')
plt.grid(True, axis='y')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate percentage change in mean, std for ModA and ModB
mean_moda_before = df_before['ModA'].mean()
mean_moda_after = df_after['ModA'].mean()
mean_modb_before = df_before['ModB'].mean()
mean_modb_after = df_after['ModB'].mean()

std_moda_before = df_before['ModA'].std()
std_moda_after = df_after['ModA'].std()
std_modb_before = df_before['ModB'].std()
std_modb_after = df_after['ModB'].std()

print("\nPercentage Change after Cleaning:")
print(f"ModA Mean: {((mean_moda_after - mean_moda_before) / mean_moda_before * 100):.2f}%")
print(f"ModA Std Dev: {((std_moda_after - std_moda_before) / std_moda_before * 100):.2f}%")
print(f"ModB Mean: {((mean_modb_after - mean_modb_before) / mean_modb_before * 100):.2f}%")
print(f"ModB Std Dev: {((std_modb_after - std_modb_before) / std_modb_before * 100):.2f}%")

# Correlation 
 #1. Correlation Heatmap
print("\nCalculating correlations between key variables...")
correlation_cols = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
correlation_matrix = df_soeraleone_clean[correlation_cols].corr()

# Turn off interactive mode for more reliable display
plt.ioff()

# Create the figure and plot
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
plt.title('Correlation Heatmap of Key Variables')
plt.tight_layout()

# Force the plot to render and display - using block=True ensures it appears
print("Displaying correlation heatmap. Close the figure window to continue...")
plt.show(block=True)

# 2. Scatter plots for wind vs. solar irradiance
plt.figure(figsize=(18, 6))

# Wind Speed vs GHI
plt.subplot(1, 3, 1)
plt.scatter(df_soeraleone_clean['WS'], df_soeraleone_clean['GHI'], alpha=0.5, color='blue')
plt.title('Wind Speed vs. GHI')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('GHI (W/m²)')
plt.grid(True)

# Wind Gust vs GHI
plt.subplot(1, 3, 2)
plt.scatter(df_soeraleone_clean['WSgust'], df_soeraleone_clean['GHI'], alpha=0.5, color='green')
plt.title('Wind Gust vs. GHI')
plt.xlabel('Wind Gust (m/s)')
plt.ylabel('GHI (W/m²)')
plt.grid(True)

# Wind Direction vs GHI
plt.subplot(1, 3, 3)
plt.scatter(df_soeraleone_clean['WD'], df_soeraleone_clean['GHI'], alpha=0.5, color='purple')
plt.title('Wind Direction vs. GHI')
plt.xlabel('Wind Direction (degrees)')
plt.ylabel('GHI (W/m²)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 3. Scatter plots for humidity vs. temperature and irradiance
plt.figure(figsize=(12, 6))

# RH vs Tamb
plt.subplot(1, 2, 1)
plt.scatter(df_soeraleone_clean['RH'], df_soeraleone_clean['Tamb'], alpha=0.5, color='red')
plt.title('Relative Humidity vs. Ambient Temperature')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Ambient Temperature (°C)')
plt.grid(True)

# RH vs GHI
plt.subplot(1, 2, 2)
plt.scatter(df_soeraleone_clean['RH'], df_soeraleone_clean['GHI'], alpha=0.5, color='orange')
plt.title('Relative Humidity vs. GHI')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('GHI (W/m²)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Wind & Distribution Analysis
# 1. Wind Rose Plot (using a polar histogram)
print("\nAnalyzing wind patterns...")

# Turn off interactive mode
plt.ioff()

# Create figure for wind rose
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')

# Convert wind direction to radians and bin the data
wd_rad = np.radians(df_soeraleone_clean['WD'])
ws_bins = [0, 2, 4, 6, 8, 10, 12, np.inf]  # Wind speed bins
wd_bins = np.linspace(0, 2*np.pi, 16)      # Direction bins (16 directions)

# Calculate histograms
hist, wd_edges, ws_edges = np.histogram2d(
    wd_rad, df_soeraleone_clean['WS'],
    bins=[wd_bins, ws_bins]
)

# Prepare for polar plot
wd_centers = (wd_edges[:-1] + wd_edges[1:]) / 2
width = 2*np.pi / (len(wd_centers))

# Create a colormap for wind speed ranges
colors = plt.cm.viridis(np.linspace(0, 1, len(ws_bins)-1))

# Plot each wind speed bin as a stacked bar
bottom = np.zeros(len(wd_centers))
for i in range(len(ws_edges)-2, -1, -1):  # Plot highest wind speeds first
    vals = hist[:, i]
    ax.bar(
        wd_centers, vals, width=width, bottom=bottom,
        color=colors[i], edgecolor='white', alpha=0.8,
        label=f'{ws_edges[i]}-{ws_edges[i+1]} m/s'
    )
    bottom += vals

# Set the compass directions
ax.set_theta_direction(-1)  # clockwise
ax.set_theta_zero_location('N')
ax.set_thetagrids(
    np.arange(0, 360, 45),
    labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
)

plt.title('Wind Rose - Speed and Direction Distribution', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Save the wind rose plot
wind_rose_file = 'output/wind_rose.png'
plt.savefig(wind_rose_file, dpi=300, bbox_inches='tight')
print(f"Wind rose plot saved to {wind_rose_file}")

# Display the wind rose plot
print("Displaying wind rose plot. Close the figure window to continue...")
plt.show(block=True)

# 2. Distribution Histograms
plt.figure(figsize=(14, 6))

# GHI Distribution
plt.subplot(1, 2, 1)
plt.hist(df_soeraleone_clean['GHI'], bins=30, color='orange', edgecolor='black', alpha=0.7)
plt.axvline(df_soeraleone_clean['GHI'].mean(), color='red', linestyle='dashed', linewidth=2)
plt.text(df_soeraleone_clean['GHI'].mean()*1.1, plt.ylim()[1]*0.9, 
         f'Mean: {df_soeraleone_clean["GHI"].mean():.1f}', color='red')
plt.title('Distribution of GHI')
plt.xlabel('GHI (W/m²)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Wind Speed Distribution
plt.subplot(1, 2, 2)
plt.hist(df_soeraleone_clean['WS'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(df_soeraleone_clean['WS'].mean(), color='blue', linestyle='dashed', linewidth=2)
plt.text(df_soeraleone_clean['WS'].mean()*1.1, plt.ylim()[1]*0.9, 
         f'Mean: {df_soeraleone_clean["WS"].mean():.1f}', color='blue')
plt.title('Distribution of Wind Speed')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the distribution histograms
hist_file = 'output/wind_ghi_histograms.png'
plt.savefig(hist_file, dpi=300, bbox_inches='tight')
print(f"Distribution histograms saved to {hist_file}")

# Display the distribution histograms
print("Displaying distribution histograms. Close the figure window to continue...")
plt.show(block=True)

# 3. Joint Distribution of GHI and Wind Speed
plt.figure(figsize=(10, 8))
sns.jointplot(
    x='WS', y='GHI', data=df_soeraleone_clean, 
    kind='hex', height=8, cmap='viridis',
    marginal_kws=dict(bins=20, fill=True)
)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.suptitle('Joint Distribution of Wind Speed and GHI', fontsize=14)
plt.show()

# Temperature Analysis
print("\nAnalyzing temperature relationships with humidity and solar radiation...")

# 1. Create RH bins for analysis
df_soeraleone_clean['RH_bin'] = pd.cut(df_soeraleone_clean['RH'], 
                                 bins=[0, 20, 40, 60, 80, 100],
                                 labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])

# 2. Plot temperature vs solar radiation colored by RH
plt.figure(figsize=(14, 10))

# Tamb vs GHI colored by RH
plt.subplot(2, 2, 1)
scatter = plt.scatter(df_soeraleone_clean['GHI'], df_soeraleone_clean['Tamb'], 
                      c=df_soeraleone_clean['RH'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Relative Humidity (%)')
plt.title('Ambient Temperature vs GHI by Humidity')
plt.xlabel('GHI (W/m²)')
plt.ylabel('Ambient Temperature (°C)')
plt.grid(True, alpha=0.3)

# 3. Average temperature by time of day and RH group
daily_temp_by_rh = df_soeraleone_clean.groupby(['Hour', 'RH_bin'])['Tamb'].mean().unstack()

plt.subplot(2, 2, 2)
daily_temp_by_rh.plot(marker='o', ax=plt.gca())
plt.title('Ambient Temperature by Hour of Day and Humidity')
plt.xlabel('Hour of Day')
plt.ylabel('Average Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.legend(title='RH Range')

# 4. Average GHI by time of day and RH group
daily_ghi_by_rh = df_soeraleone_clean.groupby(['Hour', 'RH_bin'])['GHI'].mean().unstack()

plt.subplot(2, 2, 3)
daily_ghi_by_rh.plot(marker='o', ax=plt.gca())
plt.title('Solar Irradiance (GHI) by Hour of Day and Humidity')
plt.xlabel('Hour of Day')
plt.ylabel('Average GHI (W/m²)')
plt.grid(True, alpha=0.3)
plt.legend(title='RH Range')

# 5. Module temperature difference from ambient by RH
df_soeraleone_clean['TempDiff_A'] = df_soeraleone_clean['TModA'] - df_soeraleone_clean['Tamb']
df_soeraleone_clean['TempDiff_B'] = df_soeraleone_clean['TModB'] - df_soeraleone_clean['Tamb']

plt.subplot(2, 2, 4)
rh_bins = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
rh_counts = df_soeraleone_clean['RH_bin'].value_counts().reindex(rh_bins)
tempdiff_by_rh = df_soeraleone_clean.groupby('RH_bin')[['TempDiff_A', 'TempDiff_B']].mean().reindex(rh_bins)

tempdiff_by_rh.plot(kind='bar', ax=plt.gca())
plt.title('Module Temperature Difference from Ambient by Humidity')
plt.xlabel('Relative Humidity Range')
plt.ylabel('Average Temp Difference (°C)')
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)

for i, bin_name in enumerate(rh_bins):
    if bin_name in rh_counts:
        plt.text(i, 0.2, f'n={rh_counts[bin_name]}', ha='center', va='bottom', 
                 color='black', fontsize=8)

plt.tight_layout()
plt.show()

# 6. 3D Surface Plot of Temperature, Humidity and Solar Radiation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Create a sample to avoid overcrowding the plot
sample_size = min(2000, len(df_soeraleone_clean))
df_sample = df_soeraleone_clean.sample(sample_size, random_state=42)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create the scatter plot
scatter = ax.scatter(df_sample['RH'], df_sample['GHI'], df_sample['Tamb'],
                     c=df_sample['Tamb'], cmap='plasma', s=30, alpha=0.7)

# Add a color bar
cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Temperature (°C)')

# Add labels
ax.set_xlabel('Relative Humidity (%)')
ax.set_ylabel('GHI (W/m²)')
ax.set_zlabel('Ambient Temperature (°C)')
ax.set_title('3D Relationship: Humidity, Solar Radiation, and Temperature')

# Rotate the plot for better visibility
ax.view_init(30, 45)

plt.tight_layout()
plt.show()

# Bubble Chart Visualization
print("\nCreating bubble chart to visualize GHI, Temperature, and Humidity relationships...")

plt.figure(figsize=(14, 10))

# Sample the data to avoid overcrowding (if dataset is large)
sample_size = min(1000, len(df_soeraleone_clean))
df_sample = df_soeraleone_clean.sample(sample_size, random_state=42)

# Normalize bubble sizes for better visualization (RH ranges from 0-100%)
# Scale RH to reasonable bubble sizes
size_factor = 0.5  # Adjust this to change the overall bubble size
bubble_sizes = df_sample['RH'] * size_factor

# Create main bubble chart - GHI vs Tamb with bubble size = RH
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    df_sample['GHI'], 
    df_sample['Tamb'],
    s=bubble_sizes,  # Size represents RH
    c=df_sample['RH'],  # Color also represents RH for emphasis
    cmap='Blues',
    alpha=0.6,
    edgecolors='gray',
    linewidths=0.5
)

plt.title('GHI vs Temperature with Humidity (Bubble Size & Color)')
plt.xlabel('GHI (W/m²)')
plt.ylabel('Ambient Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.colorbar(scatter, label='Relative Humidity (%)')

# Add best fit line
z = np.polyfit(df_sample['GHI'], df_sample['Tamb'], 1)
p = np.poly1d(z)
x_range = np.linspace(df_sample['GHI'].min(), df_sample['GHI'].max(), 100)
plt.plot(x_range, p(x_range), "r--", alpha=0.8)

# Add a legend for bubble sizes
for rh in [20, 50, 80]:
    plt.scatter([], [], s=rh*size_factor, c='blue', alpha=0.6, 
                label=f'RH = {rh}%', edgecolors='gray', linewidths=0.5)
plt.legend(loc='upper left', title='Bubble Size Legend')

# Create second bubble chart - GHI vs TModA vs Tamb
plt.subplot(1, 2, 2)
# Color represents module temperature, size represents ambient temperature
scatter = plt.scatter(
    df_sample['GHI'],
    df_sample['TModA'],
    s=(df_sample['Tamb'] - df_sample['Tamb'].min() + 5) * 3,  # Size based on ambient temp
    c=df_sample['RH'],  # Color represents humidity
    cmap='Blues',
    alpha=0.6,
    edgecolors='gray',
    linewidths=0.5
)

plt.title('GHI vs Module Temperature A\n(Bubble Size = Tamb, Color = RH)')
plt.xlabel('GHI (W/m²)')
plt.ylabel('Module Temperature A (°C)')
plt.grid(True, alpha=0.3)
plt.colorbar(scatter, label='Relative Humidity (%)')

# Add best fit line
z = np.polyfit(df_sample['GHI'], df_sample['TModA'], 1)
p = np.poly1d(z)
x_range = np.linspace(df_sample['GHI'].min(), df_sample['GHI'].max(), 100)
plt.plot(x_range, p(x_range), "r--", alpha=0.8)

plt.tight_layout()
plt.show()
