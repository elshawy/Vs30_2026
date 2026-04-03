import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from vs30 import model_fixed, model_terrain, sites_cluster

# Define terrain IDs
terrain_ids = {
    1: ("T01", "T01"),
    2: ("T02", "T02"),
    3: ("T03", "T03"),
    4: ("T04", "T04"),
    5: ("T05", "T05"),
    6: ("T06", "T06"),
    7: ("T07", "T07"),
    8: ("T08", "T08"),
    9: ("T09", "T09"),
    10: ("T10", "T10"),
    11: ("T11", "T11"),
    12: ("T12", "T12"),
    13: ("T13", "T13"),
    14: ("T14", "T14"),
    15: ("T15", "T15"),
    16: ("T16", "T16"),
}

# Load and preprocess data
df = pd.read_csv('V1_South.csv')
vs30_terrain_id_df = df.loc[df['tid'] != 255].copy()  # Filter out invalid GIDs
vs30_terrain_id_df = vs30_terrain_id_df.rename(columns={"NZTM_X": "easting", "NZTM_Y": "northing", "Vs30": "vs30"})
vs30_terrain_id_df['uncertainty'] = vs30_terrain_id_df['uncertainty'].round(3)

# Calculate means, errors, and counts
means, errors, counts = [], [], []
for tid, (tid_code, tid_name) in terrain_ids.items():
    count = vs30_terrain_id_df.loc[vs30_terrain_id_df['tid'] == tid].vs30.count()
    vs30_mean = vs30_terrain_id_df.loc[vs30_terrain_id_df['tid'] == tid].vs30.mean()
    vs30_std = vs30_terrain_id_df.loc[vs30_terrain_id_df['tid'] == tid].vs30.std()
    means.append(vs30_mean)
    errors.append(vs30_std)
    counts.append(count)

# Prior and posterior models
prior = model_terrain.model_prior()
prior_means = prior.T[0]
prior_errors = prior.T[1]

posterior = model_terrain.model_prior()
posterior_means = posterior.T[0]
posterior_errors = posterior.T[1]

# Updated posterior
new_posterior = model_fixed.posterior(posterior, vs30_terrain_id_df, "tid")
new_posterior_means = new_posterior.T[0]
new_posterior_errors = new_posterior.T[1]

posterior = model_terrain.model_posterior_paper()
posterior_means = posterior.T[0]
posterior_errors = posterior.T[1]
# Save prior model to CSV
prior_data = {
    "tid": list(terrain_ids.keys()),
    "tid_code": [terrain_ids[tid][0] for tid in terrain_ids.keys()],
    "tid_name": [terrain_ids[tid][1] for tid in terrain_ids.keys()],
    "prior_means": prior_means,
    "prior_errors": prior_errors,
}

# Save posterior model to CSV
posterior_data = {
    "tid": list(terrain_ids.keys()),
    "tid_code": [terrain_ids[tid][0] for tid in terrain_ids.keys()],
    "tid_name": [terrain_ids[tid][1] for tid in terrain_ids.keys()],
    "posterior_means": posterior_means,
    "posterior_errors": posterior_errors,
}

# Save updated posterior model to CSV
updated_posterior_data = {
    "tid": list(terrain_ids.keys()),
    "tid_code": [terrain_ids[tid][0] for tid in terrain_ids.keys()],
    "tid_name": [terrain_ids[tid][1] for tid in terrain_ids.keys()],
    "new_posterior_means": new_posterior_means,
    "new_posterior_errors": new_posterior_errors,
}

# Combine all CSV files into one
combined_csv_path = Path("combined_model_vs30_TID.csv")
combined_data = pd.concat([
    pd.DataFrame(prior_data),
    pd.DataFrame(posterior_data),
    pd.DataFrame(updated_posterior_data)
], axis=1)
combined_data.to_csv(combined_csv_path, index=False)

# Plotting
plt.figure(figsize=(10, 6))
scatter_label = 'Updated Vs30 Data \n (Q1:Purple, Q2:Green, Q3:Yellow)'
color_map = {0.1: 'purple', 0.2: 'green', 0.5: 'yellow'}

for i, (tid, tid_name) in enumerate(terrain_ids.items()):
    subset = vs30_terrain_id_df[vs30_terrain_id_df['tid'] == tid].copy()
    random_offsets = np.random.rand(len(subset)) * 0.2
    x_values = i + random_offsets - 0.5
    colors = subset['uncertainty'].map(color_map).fillna('gray')

    subset_q5 = subset[subset['q'] == 5]
    x_values_q5 = x_values[subset['q'] == 5]
    plt.scatter(x_values_q5, subset_q5['vs30'], c=colors[subset['q'] == 5], s=1, edgecolor='k', alpha=0.6)

    subset_not_q5 = subset[subset['q'] != 5]
    x_values_not_q5 = x_values[subset['q'] != 5]
    plt.scatter(x_values_not_q5, subset_not_q5['vs30'], c=colors[subset['q'] != 5], s=20, edgecolor='k', alpha=0.6, label=scatter_label if i == 0 else None)

# Error bars for prior, posterior, and updated posterior
prior_means_plus_1std = prior_means * (np.exp(prior.T[1]) - 1)
prior_means_minus_1std = prior_means * (1 - np.exp(-prior.T[1]))
plt.errorbar(np.arange(len(prior_means)) - 0.8, prior_means, yerr=[prior_means_minus_1std, prior_means_plus_1std], fmt='o', capsize=5, label='Median ± 1 Std (Prior)', color='darkorange')

means_plus_1std = posterior_means * (np.exp(posterior.T[1]) - 1)
means_minus_1std = posterior_means * (1 - np.exp(-posterior.T[1]))
plt.errorbar(np.arange(len(posterior_means)) - 0.6, posterior_means, yerr=[means_minus_1std, means_plus_1std], fmt='o', capsize=5, label='Median ± 1 Std (Foster et al. (2019))', color='blue')

means_plus_1std_new = new_posterior_means * (np.exp(new_posterior.T[1]) - 1)
means_minus_1std_new = new_posterior_means * (1 - np.exp(-new_posterior.T[1]))
plt.errorbar(np.arange(len(new_posterior_means)) - 0.4, new_posterior_means, yerr=[means_minus_1std_new, means_plus_1std_new], fmt='o', capsize=5, label='Median ± 1 Std (Update Result)', color='r')

plt.ylabel(r'$V_{s30} [m/s]$', fontsize=13)
xtick_labels = [f'{tid_code} ' for tid_code, _ in terrain_ids.values()]
plt.xticks(ticks=np.arange(len(xtick_labels)), labels=xtick_labels, fontsize=11, rotation=30, ha='right')
plt.yscale('log')
plt.legend()
plt.ylim(0, 1800)
plt.yticks([200, 400, 600, 800, 1000, 1200, 1400, 1600], fontsize=13)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}'.format(x) for x in current_values])
plt.grid(True)

# Add text annotations for the number of data points
y_max = 1700
for i, count in enumerate(counts):
    plt.annotate(f'{count}\n', (i - 0.5, y_max), ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('Updated_Tid_newupdate.png', dpi=600)
plt.show()

