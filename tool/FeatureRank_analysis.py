import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


file_path = "D:/peptideML_v2_0714/data/bind/end/mlData/Boruta-featureRank-XGB.csv"
output_image_path = "D:/peptideML_v2_0714/data/bind/end/else/feature_rank_boxplot.png"

max_rank = 150   # 前 N 名


# color
custom_palette = {
    "T5": "#00FF99",
    "ESM": "#9966FF",
    "Sequence-based Feature": "#33CCFF"
}
sns.set_style("ticks")

# read data
df = pd.read_csv(file_path)
df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

# selected
df_selected = df[df["rank"] <= max_rank].copy()

# feature
def classify_feature(name):
    if re.match(r'^T5_', name):
        return 'T5'
    elif re.match(r'^ESM_', name):
        return 'ESM'
    else:
        return 'Sequence-based Feature'

df_selected['group'] = df_selected['feature name'].apply(classify_feature)

# draw
plt.figure(figsize=(13.5, 9))
sns.boxplot(
    data=df_selected,
    x='group', y='rank',
    palette=custom_palette,
    width=0.6, linewidth=3.5,
    order=['Sequence-based Feature', 'T5', 'ESM']
)

plt.title(f"Feature subset (rank = {max_rank})", fontsize=40, weight='bold')
plt.xlabel('')
plt.ylabel('Feature Rank', fontsize=35)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(False)
plt.tight_layout()
plt.tick_params(direction='in', width=2, axis='both', which='major', labelsize=25, pad=20)

# line width
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

# img
plt.savefig(output_image_path, dpi=300)
plt.show()

# else
summary = df_selected.groupby('group')['rank'].describe()
print(summary)

counts = df_selected['group'].value_counts()
print("\nFeature count by group:")
print(counts)
