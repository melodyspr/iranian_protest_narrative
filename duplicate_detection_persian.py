
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re
import unicodedata
from collections import defaultdict
import random


def clean_tweet(text):
    """
    Cleans a tweet by removing URLs, mentions, hashtags, and extra spaces.
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'#\w+', '', text)    # Remove hashtags
    return ' '.join(text.split()).strip()  # Remove extra whitespace



def process_chunk(args):
    """Process a chunk of tweets to find duplicates within the chunk."""
    chunk_df, threshold = args
    groups = {}
    tweet_list = chunk_df['tweet_clean'].tolist()
    indices = chunk_df.index.tolist()
    
    for i in range(len(tweet_list)):
        if indices[i] in groups:
            continue
            
        current_tweet = tweet_list[i]
        current_length = len(current_tweet)
        group_members = [indices[i]]  # Start group with current tweet's index
        
        for j in range(i + 1, len(tweet_list)):
            if indices[j] not in groups:
                other_tweet = tweet_list[j]
                # More strict length difference
                if abs(len(other_tweet) - current_length) <= (0.3 * current_length):
                    # Use ratio instead of partial_ratio
                    score = fuzz.ratio(current_tweet, other_tweet)
                    if score >= threshold:
                        group_members.append(indices[j])
        
        if len(group_members) > 1:
            # Use the smallest index in the group as temporary group ID
            group_id = min(group_members)
            for idx in group_members:
                groups[idx] = group_id
    
    return groups

def get_median_representative(group_indices, df):
    """Select the tweet closest to median length as representative."""
    lengths = df.loc[group_indices, 'tweet_clean'].str.len()
    median_length = lengths.median()
    closest_idx = lengths.sub(median_length).abs().idxmin()
    return closest_idx

def compare_chunks(all_groups, df, threshold):
    """Compare groups across chunks to find similar groups."""
    # First collect all groups and their representative tweets
    group_reps = {}
    group_members = defaultdict(list)
    
    # First pass: collect all members of each group
    for idx, group_id in all_groups.items():
        group_members[group_id].append(idx)
    
    # Second pass: select median-length representative for each group
    for group_id, members in group_members.items():
        rep_idx = get_median_representative(members, df)
        group_reps[group_id] = (rep_idx, df.at[rep_idx, 'tweet_clean'])
    
    # Convert to lists for easier processing
    group_ids = list(group_reps.keys())
    group_rep_indices = [group_reps[gid][0] for gid in group_ids]
    group_rep_tweets = [group_reps[gid][1] for gid in group_ids]
    
    # Initialize union-find structure
    parent = {gid: gid for gid in group_ids}
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    # Compare all groups to each other
    for i in tqdm(range(len(group_ids)), desc="Comparing groups across chunks"):
        current_tweet = group_rep_tweets[i]
        current_length = len(current_tweet)
        
        for j in range(i + 1, len(group_ids)):
            other_tweet = group_rep_tweets[j]
            
            # Skip if length difference is too large
            if abs(len(other_tweet) - current_length) > (0.3 * current_length):
                continue
                
            # Use ratio instead of partial_ratio
            score = fuzz.ratio(current_tweet, other_tweet)
            if score >= threshold:
                root1 = find(group_ids[i])
                root2 = find(group_ids[j])
                if root1 != root2:
                    parent[max(root1, root2)] = min(root1, root2)  # Union by smaller ID
    
    # Apply the final group assignments
    final_groups = {}
    for idx, group_id in all_groups.items():
        final_groups[idx] = find(group_id)
    
    return final_groups

def relabel_groups(df):
    """Relabel groups sequentially from 0 to N-1, ignoring -1."""
    # Get all unique group IDs (excluding -1)
    unique_groups = sorted(df[df['group'] != -1]['group'].unique())
    
    # Create mapping from old group IDs to sequential numbers
    group_mapping = {gid: i for i, gid in enumerate(unique_groups)}
    group_mapping[-1] = -1  # Keep non-duplicates as -1
    
    # Apply the mapping
    df['group'] = df['group'].map(group_mapping)
    return df

def parallel_find_duplicates(df, threshold=84, chunk_size=100000):
    """Parallelized duplicate detection with proper cross-chunk comparison."""
    print("Preprocessing tweets...")
    df = df.copy()
    df['tweet_clean'] = df['tweet'].progress_apply(clean_tweet)
    df['word_count'] = df['tweet_clean'].str.split().str.len()

    # Separate tweets with >=5 words (for processing) and <5 words (to add back later)
    mask = df['word_count'] >= 5
    df_processed = df[mask].copy()
    df_short = df[~mask].copy()

    df_processed = df_processed.reset_index(drop=True)
    
    print("Processing in parallel chunks...")
    chunks = [(df_processed.iloc[i:i+chunk_size], threshold) 
              for i in range(0, len(df_processed), chunk_size)]
    
    with Pool(processes=cpu_count()) as pool:
        chunk_results = list(tqdm(pool.imap(process_chunk, chunks, chunksize=1), 
                            total=len(chunks), desc="Processing chunks"))
    
    # Combine all groups from chunks
    all_groups = {}
    for chunk_group in chunk_results:
        all_groups.update(chunk_group)
    
    print("Comparing groups across chunks...")
    final_groups = compare_chunks(all_groups, df_processed, threshold)
    
    # Apply final groups to dataframe
    df_processed['group'] = -1  # Initialize all as non-duplicates
    for idx, group_id in final_groups.items():
        df_processed.at[idx, 'group'] = group_id
    
    # Filter single-occurrence groups
    group_counts = df_processed['group'].value_counts()
    single_groups = group_counts[group_counts == 1].index
    df_processed.loc[df_processed['group'].isin(single_groups), 'group'] = -1
    
    print("Relabeling groups sequentially...")
    df_processed = relabel_groups(df_processed)
    
    # Add back the short tweets with group = -1
    df_short['group'] = -1
    
        # Combine the processed and short tweets
    final_df = pd.concat([df_processed, df_short], ignore_index=True)
    
    final_df = final_df.sort_values(by="created_at").reset_index(drop=True)
    
    return final_df


if __name__ == "__main__":
    tqdm.pandas()
    
    print("Loading dataset...")
    df = pd.read_pickle("../../data/persian_english_tweets_onehashtag_twomonths_processed.pkl")
    df = df[df["lang"] == "fa"].reset_index(drop=True)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values(by="created_at")
    print("\nFinding duplicates with parallel processing...")
    grouped_df = parallel_find_duplicates(df, threshold=84)
    
    print("\nSaving results...")
    grouped_df.to_csv("data/fa_duplicates_chunks_averagerep.csv", index=False)
    
    unique_groups = grouped_df[grouped_df['group'] != -1]['group'].nunique()
    duplicate_count = len(grouped_df[grouped_df['group'] != -1])
    print(f"Done! Found {unique_groups} duplicate groups ({duplicate_count} tweets, {(duplicate_count/len(grouped_df))*100:.2f}% of total)")