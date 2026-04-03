import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from vs30 import model_fixed, model_geology, sites_cluster

# Define geological IDs
geo_ids = {
    1: ("G01", "Peat"),
    2: ("G04", "Artificial fill"),
    3: ("G05", "Fluvial and estuarine deposits"),
    4: ("G06", "Alluvium and valley sediments"),
    5: ("G08", "Lacustrine"),
    6: ("G09", "Beach, bar, dune deposits"),
    7: ("G10", "Fan deposits"),
    8: ("G11", "Loess"),
    9: ("G12", "Glacigenic sediments"),
    10: ("G13", "Flood deposits"),
    11: ("G14", "Glacial moraines and till"),
    12: ("G15", "Undifferentiated sediments and sedimentary rocks"),
    13: ("G16", "Terrace deposits and old alluvium"),
    14: ("G17", "Volcanic rocks and deposits"),
    15: ("G18", "Crystalline rocks"),
}

# Load and preprocess data
df = pd.read_csv('V1_newgeo_South.csv')
vs30_geo_id_df = df.loc[(df['gid'] != 255) & (df['gid'] != 0)].copy()  # Remove invalid GIDs
vs30_geo_id_df = vs30_geo_id_df.rename(columns={"NZTM_X": "easting", "NZTM_Y": "northing", "Vs30": "vs30"})
vs30_geo_id_df['uncertainty'] = vs30_geo_id_df['uncertainty'].round(3)

# Calculate means, errors, and counts
means, errors, counts = [], [], []
for gid, (gid_code, gid_name) in geo_ids.items():
    count = vs30_geo_id_df.loc[vs30_geo_id_df['gid'] == gid].vs30.count()
    vs30_mean = vs30_geo_id_df.loc[vs30_geo_id_df['gid'] == gid].vs30.mean()
    vs30_std = vs30_geo_id_df.loc[vs30_geo_id_df['gid'] == gid].vs30.std()
    means.append(vs30_mean)
    errors.append(vs30_std)
    counts.append(count)

# Prior and posterior models
prior = model_geology.model_prior()
prior_means = prior.T[0]
prior_errors = prior.T[1]

posterior = model_geology.model_prior()
posterior_means = posterior.T[0]
posterior_errors = posterior.T[1]

# Updated posterior
new_posterior = model_fixed.posterior(posterior, vs30_geo_id_df, "gid")
new_posterior_means = new_posterior.T[0]
new_posterior_errors = new_posterior.T[1]

posterior = model_geology.model_posterior_paper()
posterior_means = posterior.T[0]
posterior_errors = posterior.T[1]

# Save prior model to CSV
prior_data = {
    "gid": list(geo_ids.keys()),
    "gid_code": [geo_ids[gid][0] for gid in geo_ids.keys()],
    "gid_name": [geo_ids[gid][1] for gid in geo_ids.keys()],
    "prior_means": prior_means,
    "prior_errors": prior_errors,
}

# Save posterior model to CSV
posterior_data = {
    "gid": list(geo_ids.keys()),
    "gid_code": [geo_ids[gid][0] for gid in geo_ids.keys()],
    "gid_name": [geo_ids[gid][1] for gid in geo_ids.keys()],
    "posterior_means": posterior_means,
    "posterior_errors": posterior_errors,
}


# Save updated posterior model to CSV
updated_posterior_data = {
    "gid": list(geo_ids.keys()),
    "gid_code": [geo_ids[gid][0] for gid in geo_ids.keys()],
    "gid_name": [geo_ids[gid][1] for gid in geo_ids.keys()],
    "new_posterior_means": new_posterior_means,
    "new_posterior_errors": new_posterior_errors,
}

# Combine all CSV files into one
combined_csv_path = Path("combined_model_vs30_gid.csv")
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

for i, (gid, gid_name) in enumerate(geo_ids.items()):
    subset = vs30_geo_id_df[vs30_geo_id_df['gid'] == gid]
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

# Final plot adjustments
plt.ylabel(r'$V_{s30} [m/s]$', fontsize=13)
xtick_labels = [f'{gid_code} ' for gid_code, _ in geo_ids.values()]
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
plt.savefig('Updated_Gid_newupdate.png', dpi=600)
plt.show()
