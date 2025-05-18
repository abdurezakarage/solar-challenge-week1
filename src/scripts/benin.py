import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
import os

# Configure matplotlib for better display of plots
matplotlib.use('TkAgg')  # Use TkAgg backend which is more reliable for showing plots
matplotlib.rcParams['figure.figsize'] = (10, 6)  # Default figure size
matplotlib.rcParams['font.size'] = 12  # Default font size
plt.ion()  # Turn on interactive mode
plt.rcParams['figure.max_open_warning'] = 50  # Allow more figures

# Print backend info for debugging
print(f"Using matplotlib backend: {matplotlib.get_backend()}")

if __name__ == "__main__":
   
    df_benin = pd.read_csv("data/benin-malanville.csv")
    df_benin['Timestamp'] = pd.to_datetime(df_benin['Timestamp'])
    #summary stats
    df_benin.describe()
     #Missing values analysis
    benin_missing_values = df_benin.isnull().sum()
    benin_missing_values = benin_missing_values[benin_missing_values > 0]
    #print(benin_missing_values)
    # Comment columns empty so we drop them
    df_benin = df_benin.drop(columns=['Comments'])
    # Drop the 'comments' column if it exists
    if 'Comments' in df_benin.columns:
        df_benin = df_benin.drop(columns=['Comments'])
        "Dropped 'Comments' column"
    else:
        "No 'Comments' column found in dataset"
    # visualize distribution of data
key_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust','Precipitation','TModA','TModB']

plt.figure(figsize=(14, 12))
for i, col in enumerate(key_cols):
    plt.subplot(4, 3, i+1)
    sns.boxplot(data=df_benin, x=col)
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
#plt.show()
# ------------------------------
# Compute Z-scores to Detect Outliers
z_scores = df_benin[key_cols].apply(zscore)
# Flag rows where |Z| > 3
benin_outlier_mask = z_scores.abs() > 3
benin_outlier_counts = benin_outlier_mask.sum()


# total number of rows flagged as outliers in any key column
benin_outlier_mask.sum()
#create graph for total outlier in the data
plt.figure(figsize=(10, 6))
sns.barplot(x=benin_outlier_counts.index, y=benin_outlier_counts.values)
plt.title('Total Outliers in Each Column')
plt.xlabel('Columns')
plt.ylabel('Number of Outliers')
plt.tight_layout()
#plt.show()

# Find all rows that have any outliers in key columns
rows_with_outliers = benin_outlier_mask.any(axis=1)

# Drop the outlier rows
df_benin_clean = df_benin[~rows_with_outliers]
#print(f"Total rows after dropping outliers: {len(df_benin_clean)}")

# Verify the data is clean by checking for outliers again
z_scores_clean = df_benin_clean[key_cols].apply(zscore)
benin_clean_outlier_mask = z_scores_clean.abs() > 3
benin_clean_outlier_count = benin_clean_outlier_mask.any(axis=1).sum()
#print(f"Remaining outliers after cleaning: {benin_clean_outlier_count}")

# Implement iterative outlier removal until convergence
#print("\nPerforming iterative outlier removal...")
iteration = 1
max_iterations = 5
previous_count = len(df_benin_clean)

while benin_clean_outlier_count > 0 and iteration <= max_iterations:
    # Remove outliers
    df_benin_clean = df_benin_clean[~benin_clean_outlier_mask.any(axis=1)]
    
    # Recalculate Z-scores and outliers
    z_scores_clean = df_benin_clean[key_cols].apply(zscore)
    benin_clean_outlier_mask = z_scores_clean.abs() > 3
    benin_clean_outlier_count = benin_clean_outlier_mask.any(axis=1).sum()
    
    # Print results
    current_count = len(df_benin_clean)
    removed = previous_count - current_count
    # print(f"Iteration {iteration}: Removed {removed} rows, {benin_clean_outlier_count} outliers remaining")
    
    # Update for next iteration
    previous_count = current_count
    iteration += 1

#print(f"\nFinal clean dataset: {len(df_benin_clean)} rows")
#print(f"Total rows removed: {len(df_benin) - len(df_benin_clean)}")

# Save the cleaned dataset
clean_file_path = "data/benin-malanville-clean.csv"
df_benin_clean.to_csv(clean_file_path, index=False)
#print(f"Cleaned data saved to {clean_file_path}")

# # Compare data statistics before and after cleaning
#print("\nStatistics before cleaning:")
#print(df_benin[key_cols].describe().T[['mean', 'std', 'min', 'max']])

#print("\nStatistics after cleaning:")
#print(df_benin_clean[key_cols].describe().T[['mean', 'std', 'min', 'max']])

# Create a better visualization to compare before and after cleaning
plt.figure(figsize=(18, 15))

# For each statistic, create a subplot to compare before and after
stats_to_compare = ['mean', 'std', 'min', 'max']
for i, stat in enumerate(stats_to_compare):
    plt.subplot(2, 2, i+1)
    
    # Get the values before and after cleaning
    before_values = df_benin[key_cols].describe().T[stat]
    after_values = df_benin_clean[key_cols].describe().T[stat]
    
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
    sns.boxplot(data=[df_benin[col], df_benin_clean[col]])
    plt.title(f'{col} Distribution Comparison')
    plt.xticks([0, 1], ['Original', 'Cleaned'])

plt.tight_layout()
#plt.show()

# Time Series Analysis for cleaned data

df_benin_clean['Timestamp'] = pd.to_datetime(df_benin_clean['Timestamp'])
#how many columns are in the dataset
# Extract date components for grouping
df_benin_clean['Date'] = df_benin_clean['Timestamp'].dt.date
df_benin_clean['Month'] = df_benin_clean['Timestamp'].dt.month
df_benin_clean['Day'] = df_benin_clean['Timestamp'].dt.day
df_benin_clean['Hour'] = df_benin_clean['Timestamp'].dt.hour

# 1. Daily patterns - Mean values by hour of day
daily_patterns = df_benin_clean.groupby('Hour')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

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
monthly_patterns = df_benin_clean.groupby('Month')[['GHI', 'DNI', 'DHI', 'Tamb']].mean()

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
df_before = df_benin.copy()
df_before['Cleaning'] = 'Before'

# Create a copy of the cleaned data with an "After" flag
df_after = df_benin_clean.copy()
df_after['Cleaning'] = 'After'

# Combine both datasets
combined_df = pd.concat([df_before, df_after])

# Calculate module statistics before and after cleaning
module_stats = combined_df.groupby('Cleaning')[['ModA', 'ModB']].agg(['mean', 'std', 'min', 'max'])
print("\nModule statistics before and after cleaning:")
print(module_stats)

# Create visualization for ModA and ModB before and after cleaning
plt.figure(figsize=(14, 10))

# Plot ModA comparison
plt.subplot(2, 2, 1)
module_means = combined_df.groupby('Cleaning')['ModA'].mean()
module_means.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'lightgreen'])
plt.title('Impact of Cleaning on ModA (Mean)')
plt.ylabel('ModA Mean (W/m²)')
plt.grid(True, axis='y')
plt.xticks(rotation=0)

# Plot ModB comparison
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

# 1. Correlation Heatmap
print("\nCalculating correlations between key variables...")
correlation_cols = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
correlation_matrix = df_benin_clean[correlation_cols].corr()

# Make sure matplotlib is in interactive mode
plt.ion()

# Display the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Key Variables')
plt.tight_layout()
plt.show()

# Add a pause to keep the plot visible
plt.pause(0.1)
input("Press Enter to continue...")

# 2. Scatter plots for wind vs. solar irradiance
plt.figure(figsize=(18, 6))

# Wind Speed vs GHI - simple scatter
plt.subplot(1, 3, 1)
plt.scatter(df_benin_clean['WS'], df_benin_clean['GHI'], 
           s=50,  # Larger point size for visibility
           alpha=0.8,  # More opaque
           color='blue',
           marker='o')  # Circle marker
plt.title('Wind Speed vs. GHI')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('GHI (W/m²)')
plt.grid(True)

# Wind Gust vs GHI - simple scatter
plt.subplot(1, 3, 2)
plt.scatter(df_benin_clean['WSgust'], df_benin_clean['GHI'], 
           s=50,  # Larger point size for visibility
           alpha=0.8,  # More opaque
           color='green',
           marker='o')  # Circle marker
plt.title('Wind Gust vs. GHI')
plt.xlabel('Wind Gust (m/s)')
plt.ylabel('GHI (W/m²)')
plt.grid(True)

# Wind Direction vs GHI - simple scatter
plt.subplot(1, 3, 3)
plt.scatter(df_benin_clean['WD'], df_benin_clean['GHI'], 
           s=50,  # Larger point size for visibility
           alpha=0.8,  # More opaque
           color='purple',
           marker='o')  # Circle marker
plt.title('Wind Direction vs. GHI')
plt.xlabel('Wind Direction (degrees)')
plt.ylabel('GHI (W/m²)')
plt.grid(True)

plt.tight_layout()
plt.draw()  # Draw the plot
plt.pause(0.1)  # Add a small pause to ensure it renders
input("Press Enter to continue to the next plot...")  # Wait for user input
plt.show()

# 3. Scatter plots for humidity vs. temperature and irradiance
plt.figure(figsize=(12, 6))

# RH vs Tamb - simple scatter
plt.subplot(1, 2, 1)
plt.scatter(df_benin_clean['RH'], df_benin_clean['Tamb'], 
           s=50,  # Larger point size for visibility
           alpha=0.8,  # More opaque
           color='red',
           marker='o')  # Circle marker
plt.title('Relative Humidity vs. Ambient Temperature')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Ambient Temperature (°C)')
plt.grid(True)

# RH vs GHI - simple scatter
plt.subplot(1, 2, 2)
plt.scatter(df_benin_clean['RH'], df_benin_clean['GHI'], 
           s=50,  # Larger point size for visibility
           alpha=0.8,  # More opaque
           color='orange',
           marker='o')  # Circle marker
plt.title('Relative Humidity vs. GHI')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('GHI (W/m²)')
plt.grid(True)

plt.tight_layout()
plt.draw()  # Draw the plot
plt.pause(0.1)  # Add a small pause to ensure it renders
input("Press Enter to continue to the next plot...")  # Wait for user input
plt.show()
# Wind & Distribution Analysis

# 1. Wind Rose Plot (using a polar histogram)
print("\nAnalyzing wind patterns...")

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')

# Convert wind direction to radians and bin the data
wd_rad = np.radians(df_benin_clean['WD'])
ws_bins = [0, 2, 4, 6, 8, 10, 12, np.inf]  # Wind speed bins
wd_bins = np.linspace(0, 2*np.pi, 16)      # Direction bins (16 directions)

# Calculate histograms
hist, wd_edges, ws_edges = np.histogram2d(
    wd_rad, df_benin_clean['WS'],
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
plt.draw()  # Draw the plot
plt.pause(0.1)  # Add a small pause to ensure it renders
input("Press Enter to continue to the next plot...")  # Wait for user input
plt.show()

# 2. Distribution Histograms
plt.figure(figsize=(14, 6))

# GHI Distribution
plt.subplot(1, 2, 1)
n, bins, patches = plt.hist(df_benin_clean['GHI'], bins=30, color='orange', 
                           edgecolor='black', alpha=0.7, linewidth=1.2)
plt.axvline(df_benin_clean['GHI'].mean(), color='red', linestyle='dashed', linewidth=2)
mean_height = max(n) * 0.9
plt.text(df_benin_clean['GHI'].mean()*1.1, mean_height, 
         f'Mean: {df_benin_clean["GHI"].mean():.1f}', color='red', fontsize=12, fontweight='bold')
plt.title('Distribution of GHI', fontsize=14, fontweight='bold')
plt.xlabel('GHI (W/m²)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)

# Wind Speed Distribution
plt.subplot(1, 2, 2)
n, bins, patches = plt.hist(df_benin_clean['WS'], bins=20, color='skyblue', 
                           edgecolor='black', alpha=0.7, linewidth=1.2)
plt.axvline(df_benin_clean['WS'].mean(), color='blue', linestyle='dashed', linewidth=2)
mean_height = max(n) * 0.9
plt.text(df_benin_clean['WS'].mean()*1.1, mean_height, 
         f'Mean: {df_benin_clean["WS"].mean():.1f}', color='blue', fontsize=12, fontweight='bold')
plt.title('Distribution of Wind Speed', fontsize=14, fontweight='bold')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.draw()  # Draw the plot
plt.pause(0.1)  # Add a small pause to ensure it renders
input("Press Enter to continue to the next plot...")  # Wait for user input
plt.show()

# 3. Joint Distribution of GHI and Wind Speed
print("\nCreating joint distribution plot...")
plt.figure(figsize=(14, 10))

# Create a simpler version by making two linked subplots
# First create a scatter plot
plt.subplot(2, 2, (1, 3))  # Span the left column
plt.scatter(df_benin_clean['WS'], df_benin_clean['GHI'], 
           alpha=0.7, c='teal', s=40, edgecolor='darkblue', linewidth=0.5)
plt.title('Joint Distribution: Wind Speed and GHI', fontsize=14, fontweight='bold')
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('GHI (W/m²)', fontsize=12)
plt.grid(True, alpha=0.3)


# Top histogram - Wind Speed
plt.subplot(2, 2, 2)
plt.hist(df_benin_clean['WS'], bins=20, color='teal', edgecolor='darkblue', alpha=0.7)
plt.title('Wind Speed Distribution', fontsize=12)
plt.xlabel('Wind Speed (m/s)', fontsize=10)
plt.grid(True, alpha=0.3)

# Right histogram - GHI
plt.subplot(2, 2, 4)
plt.hist(df_benin_clean['GHI'], bins=30, color='teal', edgecolor='darkblue', alpha=0.7, orientation='horizontal')
plt.title('GHI Distribution', fontsize=12)
plt.ylabel('GHI (W/m²)', fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.draw()  # Draw the plot
plt.pause(0.1)  # Add a small pause to ensure it renders
input("Press Enter to continue to the next plot...")  # Wait for user input
plt.show()

# Standalone Wind Speed Distribution Histogram
print("\nCreating detailed Wind Speed distribution histogram...")
plt.figure(figsize=(12, 8))

# Create a more detailed wind speed histogram
n, bins, patches = plt.hist(df_benin_clean['WS'], bins=25, color='royalblue', 
                           edgecolor='navy', alpha=0.8, linewidth=1.5)

# Add mean and median lines
plt.axvline(df_benin_clean['WS'].mean(), color='red', linestyle='dashed', linewidth=2.5, label=f'Mean: {df_benin_clean["WS"].mean():.2f} m/s')
plt.axvline(df_benin_clean['WS'].median(), color='green', linestyle='dotted', linewidth=2.5, label=f'Median: {df_benin_clean["WS"].median():.2f} m/s')

# Add a kernel density estimate
density = stats.gaussian_kde(df_benin_clean['WS'])
x_vals = np.linspace(df_benin_clean['WS'].min(), df_benin_clean['WS'].max(), 200)
plt.plot(x_vals, density(x_vals) * len(df_benin_clean) * (bins[1] - bins[0]), 
         'k-', linewidth=2, label='Density Curve')

# Add statistics as a text box
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
stat_text = (f"Count: {len(df_benin_clean)}\n"
             f"Mean: {df_benin_clean['WS'].mean():.2f} m/s\n"
             f"Median: {df_benin_clean['WS'].median():.2f} m/s\n"
             f"Std Dev: {df_benin_clean['WS'].std():.2f} m/s\n"
             f"Min: {df_benin_clean['WS'].min():.2f} m/s\n"
             f"Max: {df_benin_clean['WS'].max():.2f} m/s")
plt.text(0.75, 0.80, stat_text, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=props)

plt.title('Detailed Wind Speed Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Wind Speed (m/s)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.draw()
plt.pause(0.1)
input("Press Enter to continue to the next plot...")
plt.show()

# Temperature Analysis
print("\nAnalyzing temperature relationships with humidity and solar radiation...")

# 1. Create RH bins for analysis
df_benin_clean['RH_bin'] = pd.cut(df_benin_clean['RH'], 
                                 bins=[0, 20, 40, 60, 80, 100],
                                 labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])

# 2. Plot temperature vs solar radiation colored by RH
plt.figure(figsize=(14, 10))

# Tamb vs GHI colored by RH
plt.subplot(2, 2, 1)
scatter = plt.scatter(df_benin_clean['GHI'], df_benin_clean['Tamb'], 
                      c=df_benin_clean['RH'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Relative Humidity (%)')
plt.title('Ambient Temperature vs GHI by Humidity')
plt.xlabel('GHI (W/m²)')
plt.ylabel('Ambient Temperature (°C)')
plt.grid(True, alpha=0.3)

# 3. Average temperature by time of day and RH group
daily_temp_by_rh = df_benin_clean.groupby(['Hour', 'RH_bin'])['Tamb'].mean().unstack()

plt.subplot(2, 2, 2)
daily_temp_by_rh.plot(marker='o', ax=plt.gca())
plt.title('Ambient Temperature by Hour of Day and Humidity')
plt.xlabel('Hour of Day')
plt.ylabel('Average Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.legend(title='RH Range')

# 4. Average GHI by time of day and RH group
daily_ghi_by_rh = df_benin_clean.groupby(['Hour', 'RH_bin'])['GHI'].mean().unstack()

plt.subplot(2, 2, 3)
daily_ghi_by_rh.plot(marker='o', ax=plt.gca())
plt.title('Solar Irradiance (GHI) by Hour of Day and Humidity')
plt.xlabel('Hour of Day')
plt.ylabel('Average GHI (W/m²)')
plt.grid(True, alpha=0.3)
plt.legend(title='RH Range')

# 5. Module temperature difference from ambient by RH
df_benin_clean['TempDiff_A'] = df_benin_clean['TModA'] - df_benin_clean['Tamb']
df_benin_clean['TempDiff_B'] = df_benin_clean['TModB'] - df_benin_clean['Tamb']

plt.subplot(2, 2, 4)
rh_bins = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
rh_counts = df_benin_clean['RH_bin'].value_counts().reindex(rh_bins)
tempdiff_by_rh = df_benin_clean.groupby('RH_bin')[['TempDiff_A', 'TempDiff_B']].mean().reindex(rh_bins)

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
sample_size = min(2000, len(df_benin_clean))
df_sample = df_benin_clean.sample(sample_size, random_state=42)

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
sample_size = min(1000, len(df_benin_clean))
df_sample = df_benin_clean.sample(sample_size, random_state=42)

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

# Add this at the very end of the file, after all other code
def show_wind_histogram():
    """Function to display just the wind histogram as a basic plot"""
    # Turn off interactive mode for this function
    plt.ioff()
    
    # Create a new figure with a simple histogram
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Create the histogram with basic styling
    counts, bins, patches = ax.hist(df_benin_clean['WS'].dropna(), 
                                   bins=20, 
                                   color='navy', 
                                   edgecolor='white')
    
    # Add title and labels
    ax.set_title('Wind Speed Distribution Histogram', fontsize=16)
    ax.set_xlabel('Wind Speed (m/s)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    # Add mean line
    mean_ws = df_benin_clean['WS'].mean()
    ax.axvline(mean_ws, color='red', linestyle='dashed', linewidth=2, 
              label=f'Mean: {mean_ws:.2f} m/s')
    ax.legend()
    
    # Force drawing and show with block=True to ensure it displays
    plt.tight_layout()
    fig.canvas.draw()
    # Save the figure as an image file
    try:
        print("Saving wind histogram as image...")
        plt.savefig('wind_speed_histogram.png')
        print(f"Saved to {os.path.abspath('wind_speed_histogram.png')}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        
    # Show the plot, blocking until window is closed
    print("Displaying Wind Speed Histogram. Close the plot window to continue.")
    plt.show(block=True)

# Call the function after all other plots
show_wind_histogram()
