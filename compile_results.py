import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the folders with metrics.txt files
# metrics_dir = "evaluation_results"
metrics_dirs = ["evaluation_results", "evaluation_results_shooter"]

# Initialize a dictionary to store the data for each type
data_dict = {
    "Augmented UE4+UE5": {},
    "Augmented UE5": {},
    "UE4+UE5": {},
    "UE5": {}
}

# Iterate through the folders
for metrics_dir in metrics_dirs:
    for folder in os.listdir(metrics_dir):
        folder_path = os.path.join(metrics_dir, folder)

        if os.path.isdir(folder_path):
            weights_name = folder.split("_conf")[0]  # Extract the weights name
            if metrics_dir.endswith("shooter"):
                conf_value = float(folder.split("_conf")[1])
            else:
                conf_gun_value = float(folder.split("_conf_gun")[1].split("_conf")[0])
                conf_value = float(folder.split("_conf")[2])

            metrics_file = os.path.join(folder_path, "metrics.txt")

            # Read the metrics.txt file into a DataFrame
            df = pd.read_csv(metrics_file, delim_whitespace=True, index_col=0)  # Include the names column as the index

            # Get the "OVERALL" row and add it to the data_dict
            overall_row = df.loc["OVERALL"]

            # Add the conf_gun value to the overall_row dictionary
            overall_row["conf"] = conf_value

            # Determine the type of data based on the prefix
            if folder.startswith("exp_yolov8n_AugC") or folder.startswith("exp_yolov8n_CMasked_AugC"):
                data_type = "Augmented UE4+UE5"
            elif folder.startswith("exp_yolov8n_Aug") or folder.startswith("exp_yolov8n_Masked_Aug"):
                data_type = "Augmented UE5"
            elif folder.startswith("exp_yolov8n_C"):
                data_type = "UE4+UE5"
            else:
                data_type = "UE5"

            data_dict[data_type][(weights_name[12:-3], conf_value)] = overall_row

    # Create separate DataFrames for each type of data
    data_dfs = {data_type: pd.DataFrame(data).T for data_type, data in data_dict.items()}

    # Create separate plots for each type of data
    for data_type, data_df in data_dfs.items():
        if data_df.empty:
            continue
        # Sort the DataFrame by the value of conf_gun
        # data_df = data_df.sort_values(by="conf")
        data_df = data_df.sort_values(by="Rcll")
        # data_df = data_df.sort_values(by="Prcn")

        # Group data by weights name
        grouped = data_df.groupby(level=0)

        # Additional colors
        colormap = plt.cm.get_cmap("tab20")

        # Plotting Precision-Recall
        plt.figure(figsize=(10, 6))
        for i, (name, group) in enumerate(grouped):
            # Reset the index of the group DataFrame
            group_reset = group.reset_index()

            # Get color from colormap
            color = colormap(i % colormap.N)

            # Drop rows with NaN values in either "Prcn" or "Rcll" column
            group_reset = group_reset.dropna(subset=["Prcn", "Rcll"]).apply(
                lambda x: (x.str.rstrip('%').astype(float)) / 100 if
                x.name in ["MOTA", "Prcn", "Rcll"] else x, axis=0)

            # Sort the group by Rcll values
            group_reset = group_reset.sort_values(by="Rcll")

            # Convert the columns to lists
            prcn_values = group_reset["Prcn"].tolist()
            rcll_values = group_reset["Rcll"].tolist()
            # print(prcn_values)

            # Plot the Precision-Recall curve
            plt.plot(rcll_values, prcn_values, marker='o', label=str(name), color=color)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall - {data_type}")
        plt.legend()
        plt.grid()

        # Set x-axis limits for recall
        plt.xlim(0.0, 1.0)

        # Set y-axis limits for precision
        plt.ylim(0.0, 1.0)

        # plt.show()
        if metrics_dir.endswith("shooter"):
            fname = "figures/PR_shooter_"+data_type+".png"
        else:
            fname = "figures/PR_"+data_type+".png"
        plt.savefig(fname)
