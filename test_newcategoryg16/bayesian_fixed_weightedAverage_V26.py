import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from vs30 import model_fixed_weights, model_geology, model_terrain, sites_cluster

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
    16: ("G19", "Christchurch Holocene")
}

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

def process_and_plot_model(col_name, class_ids_dict, model_module, invalid_ids, out_csv_path, out_png_path, exclude_q3=False):
    """
    Unified function to process and plot Vs30 data for either Geology (GID) or Terrain (TID).
    
    col_name: "gid" or "tid"
    class_ids_dict: geo_ids or terrain_ids
    model_module: model_geology or model_terrain
    invalid_ids: list of invalid IDs to drop (e.g., [0, 255] or [255])
    out_csv_path: output filename for CSV
    out_png_path: output filename for PNG plot
    exclude_q3: boolean indicating whether to drop Q3 data (q == 3)
    """
    
    # Load and preprocess data
    df = pd.read_csv('measured_sites_test_newg16.csv')
    
    # Filter out Q3 data if requested by the user
    if exclude_q3:
        df = df[df['q'] != 3]
        
    df_filtered = df.loc[~df[col_name].isin(invalid_ids)].copy()
    df_filtered = df_filtered.rename(columns={"NZTM_X": "easting", "NZTM_Y": "northing", "Vs30": "vs30"})
    df_filtered['uncertainty'] = df_filtered['uncertainty'].round(3)

    # Fetch Prior models first so we can use prior_means as fallback for missing data
    prior = model_module.model_prior()
    prior_means = prior.T[0]
    prior_errors = prior.T[1]

    # Calculate means, errors, counts, and weighted statistics
    means, errors, counts = [], [], []
    weighted_means, weighted_stds = [], []

    for i, (class_id, (class_code, class_name)) in enumerate(class_ids_dict.items()):
        subset = df_filtered.loc[df_filtered[col_name] == class_id].copy()
        count = subset.vs30.count()
        
        if count > 0:
            vs30_mean = subset.vs30.mean()
            vs30_std = subset.vs30.std()
            
            # Extract vs30, uncertainty, and q values
            v = subset['vs30'].values
            u = subset['uncertainty'].values
            q_vals = subset['q'].values
            
            # Calculate weight as the inverse of variance (1 / uncertainty^2)
            w = 1.0 / (u ** 2)
            
            # Apply penalty for q == 5: Treat as 0.1 data point by dividing weight by 10
            w[q_vals == 5] /= 10.0
            
            # Calculate effective count (normal points = 1.0, q=5 points = 0.1)
            eff_count = np.sum(np.where(q_vals == 5, 0.1, 1.0))
            
            # Calculate weighted average of Vs30
            w_mean = np.average(v, weights=w)
            
            # Use effective count (eff_count) instead of raw count for threshold
            if eff_count > 3:
                # If effective N >= 4: Calculate actual spatial dispersion (Weighted Sample Standard Deviation)
                ln_v = np.log(v)
                ln_w_mean = np.average(ln_v, weights=w)
                w_std = np.sqrt(np.average((ln_v - ln_w_mean)**2, weights=w))
            else:
                # If effective N <= 3: Sample size is too small to calculate a meaningful dispersion.
                # Forcefully assign a baseline standard deviation of 0.5.
                w_std = 0.5
                
        else:
            # Fallback Logic: If no data is available, set weighted mean to prior mean and std to 0.5
            vs30_mean, vs30_std = np.nan, np.nan
            w_mean = prior_means[i]
            w_std = 0.5

        means.append(vs30_mean)
        errors.append(vs30_std)
        counts.append(count)  # Keep raw count for the text annotations on the plot
        weighted_means.append(w_mean)
        weighted_stds.append(w_std)

    # Convert weighted lists to numpy arrays for vector operations during plotting
    weighted_avg = np.array(weighted_means)
    weighted_std = np.array(weighted_stds)

    # Temporary posterior to generate updated posterior
    posterior_temp = model_module.model_prior()
    new_posterior = model_fixed_weights.posterior(posterior_temp, df_filtered, col_name)
    new_posterior_means = new_posterior.T[0]
    new_posterior_errors = new_posterior.T[1]

    # Official posterior from the module
    posterior = model_module.model_posterior_paper()
    posterior_means = posterior.T[0]
    posterior_errors = posterior.T[1]

    # Save prior model to CSV (dynamically generating keys based on col_name)
    prior_data = {
        col_name: list(class_ids_dict.keys()),
        f"{col_name}_code": [class_ids_dict[cid][0] for cid in class_ids_dict.keys()],
        f"{col_name}_name": [class_ids_dict[cid][1] for cid in class_ids_dict.keys()],
        "prior_means": prior_means,
        "prior_errors": prior_errors,
    }

    # Save posterior model to CSV
    posterior_data = {
        col_name: list(class_ids_dict.keys()),
        f"{col_name}_code": [class_ids_dict[cid][0] for cid in class_ids_dict.keys()],
        f"{col_name}_name": [class_ids_dict[cid][1] for cid in class_ids_dict.keys()],
        "posterior_means": posterior_means,
        "posterior_errors": posterior_errors,
    }

    # Save updated posterior model to CSV
    updated_posterior_data = {
        col_name: list(class_ids_dict.keys()),
        f"{col_name}_code": [class_ids_dict[cid][0] for cid in class_ids_dict.keys()],
        f"{col_name}_name": [class_ids_dict[cid][1] for cid in class_ids_dict.keys()],
        "new_posterior_means": new_posterior_means,
        "new_posterior_errors": new_posterior_errors,
    }

    # Save newly calculated weighted averages and errors to CSV
    weighted_data = {
        col_name: list(class_ids_dict.keys()),
        f"{col_name}_code": [class_ids_dict[cid][0] for cid in class_ids_dict.keys()],
        f"{col_name}_name": [class_ids_dict[cid][1] for cid in class_ids_dict.keys()],
        "weighted_means": weighted_avg,
        "weighted_errors": weighted_std,
    }

    # Combine all CSV files into one
    combined_csv_path = Path(out_csv_path)
    combined_data = pd.concat([
        pd.DataFrame(prior_data),
        pd.DataFrame(posterior_data).drop(columns=[col_name, f"{col_name}_code", f"{col_name}_name"]), 
        pd.DataFrame(updated_posterior_data).drop(columns=[col_name, f"{col_name}_code", f"{col_name}_name"]),
        pd.DataFrame(weighted_data).drop(columns=[col_name, f"{col_name}_code", f"{col_name}_name"])
    ], axis=1)
    combined_data.to_csv(combined_csv_path, index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    scatter_label = 'Updated Vs30 Data \n (Q1:Purple, Q2:Green, Q3:Yellow)'
    color_map = {0.1: 'purple', 0.2: 'green', 0.5: 'yellow'}

    for i, (class_id, (class_code, class_name)) in enumerate(class_ids_dict.items()):
        subset = df_filtered[df_filtered[col_name] == class_id].copy()
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

    # Calculate error bounds for Weighted Average using log-normal assumption
    weighted_avg_plus_1std = weighted_avg * (np.exp(weighted_std) - 1)
    weighted_avg_minus_1std = weighted_avg * (1 - np.exp(-weighted_std))
    yerr_weighted = [weighted_avg_minus_1std, weighted_avg_plus_1std]

    # Plot the weighted means and errors at x-offset -0.2
    plt.errorbar(np.arange(len(weighted_avg)) - 0.2, weighted_avg, yerr=yerr_weighted, fmt='o', capsize=5, label='Weighted Avg ± 1 Std (Data)', color='k')

    # Final plot adjustments
    plt.ylabel(r'$V_{s30} [m/s]$', fontsize=13)
    xtick_labels = [f'{class_code} ' for class_code, _ in class_ids_dict.values()]
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
    plt.savefig(out_png_path, dpi=600)
    plt.show()


# =====================================================================
# Execute Processing for Both Models
# =====================================================================

if __name__ == "__main__":
    # Get user input for excluding Q3 data
    user_input = input("Exclude Q3 data? (y/n): ").strip().lower()
    exclude_q3 = (user_input == 'y')
    
    # Append suffix to filenames if Q3 is excluded
    suffix = "_noQ3" if exclude_q3 else ""

    # 1. Process Geology (GID)
    print("Processing Geology (GID) Model...")
    process_and_plot_model(
        col_name="gid",
        class_ids_dict=geo_ids,
        model_module=model_geology,
        invalid_ids=[0, 255],
        out_csv_path=f"combined_model_vs30_gid{suffix}.csv",
        out_png_path=f"Updated_Gid_newupdate{suffix}.png",
        exclude_q3=exclude_q3
    )

    # 2. Process Terrain (TID)
    print("Processing Terrain (TID) Model...")
    process_and_plot_model(
        col_name="tid",
        class_ids_dict=terrain_ids,
        model_module=model_terrain,
        invalid_ids=[255], # Typically TID only has 255 as invalid
        out_csv_path=f"combined_model_vs30_TID{suffix}.csv",
        out_png_path=f"Updated_Tid_newupdate{suffix}.png",
        exclude_q3=exclude_q3
    )
