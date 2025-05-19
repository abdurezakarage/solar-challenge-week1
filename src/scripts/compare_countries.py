import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
from scipy.stats import f_oneway, kruskal
import os
#load the three dataset for the three countries
df_ben= pd.read_csv("data/benin-malanville-clean.csv")
df_sier = pd.read_csv("data/sierraleone-bumbuna-clean.csv")
df_tog = pd.read_csv("data/togo-dapaong-clean.csv")

# Add a 'Country' column to each DataFrame
df_ben['Country'] = 'Benin'
df_sier['Country'] = 'Sierra Leone'
df_tog['Country'] = 'Togo'

# Combine all data into one DataFrame
df_all = pd.concat([df_ben, df_sier, df_tog], ignore_index=True)

# List of metrics to plot and summarize
metrics = ['GHI', 'DNI', 'DHI']

# 1. Boxplots for each metric
plt.figure(figsize=(15, 5))
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='Country', y=metric, data=df_all, palette='Set2')
    plt.title(f'Boxplot of {metric} by Country')
plt.tight_layout()
plt.show()

# 2. Summary Table
summary = df_all.groupby('Country')[metrics].agg(['mean', 'median', 'std'])
print("Summary Table (mean, median, std):")
print(summary)

# Flatten columns for better CSV readability
summary.columns = ['_'.join(col) for col in summary.columns]

# Create directory if it doesn't exist
output_dir = "comparison_results"
os.makedirs(output_dir, exist_ok=True)

# Save summary table to the directory
summary.to_csv(os.path.join(output_dir, "country_metric_summary.csv"))
print(f"Summary table saved to {os.path.join(output_dir, 'country_metric_summary.csv')}")

# Extract GHI values for each country
ghi_ben = df_ben['GHI'].dropna()
ghi_sier = df_sier['GHI'].dropna()
ghi_tog = df_tog['GHI'].dropna()

# One-way ANOVA
anova_stat, anova_p = f_oneway(ghi_ben, ghi_sier, ghi_tog)
print(f"\nOne-way ANOVA p-value for GHI: {anova_p:.4f}")

# Kruskal–Wallis test
kruskal_stat, kruskal_p = kruskal(ghi_ben, ghi_sier, ghi_tog)
print(f"Kruskal–Wallis p-value for GHI: {kruskal_p:.4f}")

# Prepare results as a string
stat_results = (
    f"One-way ANOVA p-value for GHI: {anova_p:.4f}\n"
    f"Kruskal–Wallis p-value for GHI: {kruskal_p:.4f}\n\n"
)

if anova_p < 0.05:
    stat_results += "ANOVA: There is a statistically significant difference in GHI between countries (p < 0.05).\n"
else:
    stat_results += "ANOVA: No statistically significant difference in GHI between countries (p >= 0.05).\n"

if kruskal_p < 0.05:
    stat_results += "Kruskal–Wallis: There is a statistically significant difference in GHI between countries (p < 0.05).\n"
else:
    stat_results += "Kruskal–Wallis: No statistically significant difference in GHI between countries (p >= 0.05).\n"

# Save the results to a text file in the comparison_results directory
with open(os.path.join(output_dir, "statistical_test_results.txt"), "w") as f:
    f.write(stat_results)

print(f"Statistical test results saved to {os.path.join(output_dir, 'statistical_test_results.txt')}")

# Visual Summary: Bar chart ranking countries by average GHI
plt.figure(figsize=(6, 4))
# Prepare data for bar chart
avg_ghi = summary['GHI_mean']
avg_ghi = avg_ghi.sort_values(ascending=False)
sns.barplot(x=avg_ghi.index, y=avg_ghi.values, palette='Set2')
plt.ylabel('Average GHI')
plt.xlabel('Country')
plt.title('Countries Ranked by Average GHI')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'average_ghi_bar_chart.png'))
plt.show()
print(f"Bar chart saved to {os.path.join(output_dir, 'average_ghi_bar_chart.png')}")





