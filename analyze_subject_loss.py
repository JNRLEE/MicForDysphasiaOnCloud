
# 分析每個受試者的loss平均值並排序視覺化
# Analysis of average loss for each subject and visualization with sorting

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Read the CSV file
df = pd.read_csv('ResultFig/tsne_results_with1130keras.csv')

# Extract subject ID from file path using regex
def extract_subject_id(path):
    match = re.search(r'[NP]\d{3}', path)
    return match.group(0) if match else None

# Add subject ID column
df['subject_id'] = df['file_path'].apply(extract_subject_id)

# Calculate average loss for each subject
subject_stats = df.groupby('subject_id').agg({
    'loss': ['mean', 'std', 'count'],
    'score': 'first'  # Get the score for each subject
}).reset_index()

# Flatten column names
subject_stats.columns = ['subject_id', 'mean_loss', 'std_loss', 'count', 'score']

# Sort by mean loss
subject_stats = subject_stats.sort_values('mean_loss', ascending=False)

# Create visualization
plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")

# Create bar plot
bars = plt.bar(subject_stats['subject_id'], subject_stats['mean_loss'])

# Add error bars
plt.errorbar(x=range(len(subject_stats)), 
            y=subject_stats['mean_loss'],
            yerr=subject_stats['std_loss'],
            fmt='none',
            color='black',
            capsize=5)

# Customize plot
plt.title('平均Loss值分析 (依受試者分組)\nAverage Loss Analysis by Subject', fontsize=14, pad=20)
plt.xlabel('受試者編號 Subject ID', fontsize=12)
plt.ylabel('平均Loss值 Average Loss', fontsize=12)
plt.xticks(rotation=45)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('ResultFig/subject_loss_analysis.png', dpi=300, bbox_inches='tight')

# Print statistics
print("\n受試者Loss值統計分析 Subject Loss Statistics:")
print("=" * 60)
print(f"{'Subject ID':<10} {'Mean Loss':<12} {'Std Dev':<12} {'Count':<8} {'Score':<8}")
print("-" * 60)
for _, row in subject_stats.iterrows():
    print(f"{row['subject_id']:<10} {row['mean_loss']:<12.3f} {row['std_loss']:<12.3f} {row['count']:<8.0f} {row['score']:<8.0f}")
