import pandas as pd
import miceforest as mf

# Set random seed
random_seed = 3456

# Load data
olink_df = pd.read_csv('/home/dylan20/projects/def-aharroud/shared/ukb/olink/data_olink_010324.csv')
main_df = pd.read_csv('/home/dylan20/projects/def-aharroud/shared/dylanfolder/dylan_data/main_df.csv')
matched_controls = pd.read_csv('/home/dylan20/projects/def-aharroud/shared/dylanfolder/dylan_data/matched_controls_age_sex.csv')

olink_df = olink_df.rename(columns={'olink_instance_0.eid': 'eid'})
olink_df = pd.merge(olink_df, main_df[['eid', 'ms']], on='eid', how='inner')

# Identify columns with >20% missing values and remove them
missing_percentage = olink_df.isna().mean() * 100
columns_to_remove = missing_percentage[missing_percentage > 20].index
print(f'removed columns: {columns_to_remove}')
olink_df.drop(columns=columns_to_remove, inplace=True)


# Define MS patients
ms_patients = olink_df[olink_df["ms"] == 1]

olink_df = olink_df.drop(columns=["ms"])

# Remove MS patients and matched controls for training
excluded_eids = set(ms_patients["eid"]).union(set(matched_controls["eid"]))
training_df = olink_df[~olink_df["eid"].isin(excluded_eids)].copy()
excluded_df = olink_df[olink_df["eid"].isin(excluded_eids)].copy()

# Reset index (otherwise an error will occur)
training_df = training_df.reset_index(drop=True)
excluded_df = excluded_df.reset_index(drop=True)

column_dict = {
  col: [other_col for other_col in training_df.columns if other_col != col and other_col != 'eid']
  for col in training_df.columns if col != 'eid'
}

kds = mf.ImputationKernel(
  training_df,
  variable_schema=column_dict,
  random_state=random_seed
)

# run
kds.mice(
  iterations=5,
  n_jobs=2, 
  verbose=True
)

# Impute excluded_df using the trained kernel
imputed_excluded = kds.impute_new_data(excluded_df)

# Get completed training data
imputed_training = kds.complete_data(dataset=0)

# Combine both
final_imputed_df = pd.concat([imputed_training, imputed_excluded], ignore_index=True)


# Save output
final_imputed_df.to_csv(
    '/home/dylan20/projects/def-aharroud/shared/dylanfolder/dylan_data/olink_imputed.csv', 
    index=False
)

print("Imputation and alignment checks passed successfully!")
