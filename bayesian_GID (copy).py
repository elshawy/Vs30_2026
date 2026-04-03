import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from vs30 import model_5, model_geology, sites_cluster

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

df = pd.read_csv('measured_sites.csv')
print(df)

vs30_geo_id_df = df.copy()
vs30_geo_id_df = vs30_geo_id_df.loc[vs30_geo_id_df['gid'] != 255]  # remove 255 = ID_NODATA
vs30_geo_id_df = vs30_geo_id_df.loc[vs30_geo_id_df['gid'] != 0]  # remove 0 = Water

means = []
errors = []
counts = []
for gid, (gid_code, gid_name) in geo_ids.items():
    print(gid, gid_name)
    count = vs30_geo_id_df.loc[vs30_geo_id_df['gid'] == gid].Vs30.count()
    vs30_mean = vs30_geo_id_df.loc[vs30_geo_id_df['gid'] == gid].Vs30.mean()
    vs30_std = vs30_geo_id_df.loc[vs30_geo_id_df['gid'] == gid].Vs30.std()
    print("n = {} vs30={} std= {}".format(count, vs30_mean, vs30_std))

    means.append(vs30_mean)
    errors.append(vs30_std)
    counts.append(count)

prior = model_geology.model_prior()
prior_means = prior.T[0]
prior_errors = prior.T[1] * prior_means

posterior = model_geology.model_prior()
print(posterior)
posterior_means = posterior.T[0]

means_plus_1std = posterior_means * (np.exp(posterior.T[1]) - 1)
means_minus_1std = posterior_means * (1 - np.exp(-posterior.T[1]))
yerr2 = [means_minus_1std, means_plus_1std]
print(means_plus_1std)
print(means_minus_1std)
print(yerr2)

vs30_geo_id_df = vs30_geo_id_df.rename(columns={"NZTM_X": "easting", "NZTM_Y": "northing", "Vs30": "vs30"})
new_posterior = model_5.posterior(posterior, vs30_geo_id_df, "gid")
new_posterior_means = new_posterior.T[0]
means_plus_1std_new = new_posterior_means * (np.exp(new_posterior.T[1]) - 1)
means_minus_1std_new = new_posterior_means * (1 - np.exp(-new_posterior.T[1]))
yerr = [means_minus_1std_new, means_plus_1std_new]

median_vs30 = np.median(new_posterior[:, 0])

plt.figure(figsize=(7, 6))
scatter_label = 'Updated Vs30 Data \n (Q1:Purple, Q2:Green, Q3:Yellow)'
color_map = {0.1: 'purple', 0.2: 'Green', 0.5: 'yellow'}

for i, (gid, gid_name) in enumerate(geo_ids.items()):
    subset = vs30_geo_id_df[vs30_geo_id_df['gid'] == gid]
    random_offsets = np.random.rand(len(subset)) * 0.25
    x_values = i + random_offsets

    # Map uncertainty values to colors and handle NaN values
    colors = subset['uncertainty'].map(color_map).fillna('gray')

    # Plot q == 5 first
    subset_q5 = subset[subset['q'] == 5]
    x_values_q5 = x_values[subset['q'] == 5]
    scatter_q5 = plt.scatter(x_values_q5, subset_q5['vs30'], c=colors[subset['q'] == 5], s=1, edgecolor='k', alpha=0.6)

    # Plot q != 5
    subset_not_q5 = subset[subset['q'] != 5]
    x_values_not_q5 = x_values[subset['q'] != 5]
    scatter_not_q5 = plt.scatter(x_values_not_q5, subset_not_q5['vs30'], c=colors[subset['q'] != 5], s=20, edgecolor='k', alpha=0.6,label=scatter_label if i == 0 else None)

posterior = model_geology.model_posterior_paper()
print(posterior)
posterior_means = posterior.T[0]
means_plus_1std = posterior_means * (np.exp(posterior.T[1]) - 1)
means_minus_1std = posterior_means * (1 - np.exp(-posterior.T[1]))
yerr2 = [means_minus_1std, means_plus_1std]
plt.errorbar(np.arange(len(posterior_means)) -0.1,posterior_means, yerr=yerr2, fmt='o', capsize=5, label='Median ± 1 Std (Foster et al. (2019))', color='blue')

# Plot the median values for new_posterior
plt.errorbar(np.arange(len(new_posterior_means)) + 0.4, new_posterior_means, yerr=yerr, fmt='o', capsize=5, label='Median ± 1 Std (Updated Dataset)', color='r')

posterior = model_geology.model_prior()
posterior_means = posterior.T[0]

means_plus_1std = posterior_means * (np.exp(posterior.T[1]) - 1)
means_minus_1std = posterior_means * (1 - np.exp(-posterior.T[1]))
yerr2 = [means_minus_1std, means_plus_1std]

plt.xlabel('gid', fontsize=13)
plt.ylabel(r'$V_{s30} [m/s]$', fontsize=13)
xtick_labels = ['G01', 'G04', 'G05', 'G06', 'G08', 'G09', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18']

plt.xticks(ticks=np.arange(len(new_posterior_means)) + 0.2, labels=xtick_labels, fontsize=13, rotation=45)  # Rotate x-ticks by 90 degrees
plt.legend()
plt.yscale('log')
plt.ylim(0, 1800)
plt.yticks([200, 400, 600, 800, 1000, 1200, 1400, 1600], fontsize=13)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}'.format(x) for x in current_values])
print(new_posterior)
plt.grid(True)  # Add grid lines

# Add text annotations for the number of data points
y_max = 1700  # Maximum y-value
offset = 0  # Offset to move the text above the maximum y-value
for i, count in enumerate(counts):
    plt.annotate(f'{count}\n', (i+0.25, y_max + offset), ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('Updated_Gid3_2nd_te.png', dpi=600)
plt.show()