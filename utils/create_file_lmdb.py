import os
import json
import lmdb
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Number of parallel shards / processes
N_SHARDS = 32
# Estimated total size of data per shard (in bytes), adjust if needed
MAP_SIZE_PER_SHARD = int(1e12 // N_SHARDS)

VIEWS = {
    'syncam4d': 36,
    'kubric4d': 16,
    'obj4d-10k': 18,
}

def split_and_map_ids(ids, n_shards):
    """
    Split a list of sequence IDs into n_shards batches and return:
      - id_batches: a list of lists of IDs
      - seq_to_shard: a dict mapping each seq -> its shard index
    """
    id_batches = np.array_split(ids, n_shards)
    seq_to_shard = {}
    for shard_idx, batch in enumerate(id_batches):
        for seq in batch:
            seq_to_shard[seq] = shard_idx
    return id_batches, seq_to_shard


def build_lmdb_shard(root, dataset_name, split, ids_batch, num_views, lmdb_path, map_size):
    """
    Build one LMDB shard containing only the sequences in ids_batch.
    """
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    # Open shard environment (single-writer)
    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        writemap=True,
        lock=True
    )
    with env.begin(write=True) as txn:
        for seq in tqdm(ids_batch, desc=f"Shard {os.path.basename(lmdb_path)}", unit="seq"):
            for view in range(num_views):
                npy_file = os.path.join(
                    root, 'alldata', dataset_name, split, f"{seq}_{view}.npy"
                )
                arr = np.load(npy_file)
                key = f"{seq}_{view}".encode('ascii')
                txn.put(key, arr.tobytes())
    env.sync()
    env.close()


def main():
    dataset_root = "/fs-computility/llm/shared/konglingdong/data/sets/see4d/sync4d"
    lmdb_root    = "/fs-computility/llm/shared/konglingdong/data/sets/see4d/sync4d/lmdb"
    dataset_list = ['syncam4d', 'kubric4d', 'obj4d-10k']
    splits       = ['train', 'test']

    for ds in dataset_list:
        for split in splits:
            # Paths
            index_file = os.path.join(dataset_root, ds, f"index_{split}.json")
            shard_dir  = os.path.join(lmdb_root, ds, split)
            seq_map_file = os.path.join(shard_dir, 'seq_to_shard.json')

            # Load all sequence IDs
            with open(index_file, 'r') as f:
                all_ids = json.load(f)
            num_views = VIEWS[ds]

            # 1) Split IDs and build mapping
            id_batches, seq_to_shard = split_and_map_ids(all_ids, N_SHARDS)

            # 2) Save seq_to_shard.json for dataset init
            os.makedirs(shard_dir, exist_ok=True)
            with open(seq_map_file, 'w') as f:
                json.dump(seq_to_shard, f)

            # 3) Build LMDB shards in parallel
            jobs = []
            with ProcessPoolExecutor(max_workers=N_SHARDS) as executor:
                for shard_idx, batch in enumerate(id_batches):
                    lmdb_path = os.path.join(
                        shard_dir, f"data_shard_{shard_idx}.lmdb"
                    )
                    jobs.append(
                        executor.submit(
                            build_lmdb_shard,
                            dataset_root,
                            ds,
                            split,
                            batch.tolist(),
                            num_views,
                            lmdb_path,
                            MAP_SIZE_PER_SHARD
                        )
                    )
                # Wait for all shards to finish
                for job in tqdm(jobs, desc=f"Building {ds}/{split}", unit="shard"):
                    job.result()

    print("All LMDB shards built and seq_to_shard mappings saved.")


if __name__ == "__main__":
    main()
