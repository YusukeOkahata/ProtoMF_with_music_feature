import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set format for logging data stats
INF_STR = "{:10d} entries {:7d} users {:7d} items for {}"

# Argument parser to specify paths
parser = argparse.ArgumentParser()
parser.add_argument('--listening_history_path', '-lh', type=str,
                    help="Path to the folder containing 'users.tsv', 'inter_dataset.tsv', and 'tracks.tsv'.")
parser.add_argument('--saving_path', '-s', type=str, default='./',
                    help="Path where to save the split data.")

args = parser.parse_args()
listening_history_path = args.listening_history_path
saving_path = args.saving_path
k = 1  # Leave-1-out split

# Define paths to each file
user_info_path = os.path.join(listening_history_path, 'users.tsv')
listening_events_path = os.path.join(listening_history_path, 'inter_dataset.tsv')

# Load listening events data
lhs = pd.read_csv(listening_events_path, sep='\t', names=['old_user_id', 'old_item_id', 'timestamp'], skiprows=1, usecols=[0, 1, 3])
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), 'Original Data'))

# エントリ数を1/2に削減
target_entries = len(lhs) // 2
lhs = lhs.sample(n=target_entries, random_state=42)
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), 'After entry sampling'))


# Filter data to keep only the last month (after 2020-02-20)
lhs = lhs[lhs.timestamp >= '2020-02-20 00:00:00']
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), 'Only last month'))

# Load user data and filter by age and gender
users = pd.read_csv(user_info_path, sep='\t', names=['old_user_id', 'country', 'age', 'gender', 'creation_time'], skiprows=1)
users = users[(users.gender.isin(['m', 'f'])) & (users.age >= 10) & (users.age <= 95)]
lhs = lhs[lhs.old_user_id.isin(set(users.old_user_id))]
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), 'Only users with gender and valid age'))

# Keep only the first interaction for each user-item pair
lhs = lhs.sort_values('timestamp')
lhs = lhs.drop_duplicates(subset=['old_user_id', 'old_item_id'])
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), 'Keeping only the first interaction'))

# Remove power users above the 99th percentile of interaction count
user_counts = lhs.old_user_id.value_counts()
perc_99 = np.percentile(user_counts, 99)
user_below = set(user_counts[user_counts <= perc_99].index)
lhs = lhs[lhs.old_user_id.isin(user_below)]
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), 'Removed power users (below 99% percentile)'))

# Apply 10-core filtering
while True:
    start_number = len(lhs)
    item_counts = lhs.old_item_id.value_counts()
    item_above = set(item_counts[item_counts >= 10].index)
    lhs = lhs[lhs.old_item_id.isin(item_above)]

    user_counts = lhs.old_user_id.value_counts()
    user_above = set(user_counts[user_counts >= 10].index)
    lhs = lhs[lhs.old_user_id.isin(user_above)]

    if len(lhs) == start_number:
        break
print(INF_STR.format(len(lhs), lhs.old_user_id.nunique(), lhs.old_item_id.nunique(), '10-core filtering'))

# Create integer indices for users and items
user_ids = lhs.old_user_id.drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'user_id'})
item_ids = lhs.old_item_id.drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'item_id'})
lhs = lhs.merge(user_ids).merge(item_ids)

# Leave-1-out split
lhs = lhs.sort_values('timestamp')
train_idxs, val_idxs, test_idxs = [], [], []
for user, user_group in tqdm(lhs.groupby('old_user_id')):
    if len(user_group) <= k * 2:
        train_idxs += list(user_group.index)
    else:
        train_idxs += list(user_group.index[:-2 * k])
        val_idxs += list(user_group.index[-2 * k:-k])
        test_idxs += list(user_group.index[-k:])

train_data = lhs.loc[train_idxs]
val_data = lhs.loc[val_idxs]
test_data = lhs.loc[test_idxs]



print(INF_STR.format(len(train_data), train_data.old_user_id.nunique(), train_data.old_item_id.nunique(), 'Train Data'))
print(INF_STR.format(len(val_data), val_data.old_user_id.nunique(), val_data.old_item_id.nunique(), 'Val Data'))
print(INF_STR.format(len(test_data), test_data.old_user_id.nunique(), test_data.old_item_id.nunique(), 'Test Data'))

# Save the split datasets and indices
train_data.to_csv(os.path.join(saving_path, 'listening_history_train.csv'), index=False)
val_data.to_csv(os.path.join(saving_path, 'listening_history_val.csv'), index=False)
test_data.to_csv(os.path.join(saving_path, 'listening_history_test.csv'), index=False)
user_ids.to_csv(os.path.join(saving_path, 'user_ids.csv'), index=False)
item_ids.to_csv(os.path.join(saving_path, 'item_ids.csv'), index=False)