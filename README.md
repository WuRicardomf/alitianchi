# News Recommendation Recall Pipeline

This project implements a **multi-channel news recommendation / recall system** based on click logs. It combines:

- **Item-based Collaborative Filtering (ItemCF)**
- **User-based Collaborative Filtering (UserCF)**
- **Embedding-based recall** using **YouTubeDNN**

The main entry script is `learn.py`, which builds similarity matrices, trains embedding models, and generates recall dictionaries for later ranking or evaluation.

## Project Files

| File | Description |
| --- | --- |
| `learn.py` | Main pipeline script. Loads data, builds recall channels, trains YouTubeDNN, and saves outputs. |
| `share.py` | Shared utilities for data loading, preprocessing, dictionary building, and recall evaluation. |
| `i2i_sim.py` | Item-to-item similarity computation and item-based recall functions. |
| `u2u_sim.py` | User-to-user similarity computation and user-based recall functions. |

## Data Files

The project expects the following files in the project root:

- `train_click_log.csv`
- `testA_click_log.csv`
- `articles.csv`
- `articles_emb.csv`

### Main Columns Used

#### Click logs
- `user_id`
- `click_article_id`
- `click_timestamp`

#### Article metadata
- `article_id` / `click_article_id`
- `category_id`
- `words_count`
- `created_at_ts`

#### Article embeddings
- `article_id`
- embedding feature columns such as `emb0`, `emb1`, ...

## Workflow Overview

1. Load click logs and article metadata.
2. Build user click histories and item click histories.
3. Compute:
   - item-item similarity (`ItemCF`)
   - user-user similarity (`UserCF`)
   - content embedding similarity
4. Train a `YouTubeDNN` model to obtain user and item embeddings.
5. Generate recall lists from multiple channels.
6. Save intermediate similarity matrices, embeddings, and recall results as `.pkl` files.

## How to Run

Run the main script from the project root:

```bash
python learn.py
```

### Notes

- `share.py` uses `os.getcwd()` as both `data_path` and `save_path`, so the script should be executed from the project root.
- If you want to run on a larger dataset, you can switch from sampled data to full data by changing the call in `learn.py`:
  - sampled mode: `get_all_click_sample(data_path)`
  - full mode: `get_all_click_df(data_path, offline=False)`
- `UserCF` can be memory-intensive on large datasets.

## Dependencies

Recommended Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`
- `faiss`
- `torch`
- `torch_rechub`

Install them with:

```bash
pip install pandas numpy scikit-learn tqdm faiss-cpu torch torch-rechub
```

> If you have a GPU environment, you may prefer a GPU-enabled FAISS or PyTorch build.

## Output Files

The pipeline may generate the following files:

- `emb_i2i_sim.pkl` — embedding-based item-item similarity matrix
- `itemcf_i2i_sim.pkl` — ItemCF similarity matrix
- `usercf_u2u_sim.pkl` — UserCF similarity matrix
- `item_youtube_emb.pkl` — item embeddings from YouTubeDNN
- `user_youtube_emb.pkl` — user embeddings from YouTubeDNN
- `youtube_u2i_dict.pkl` — YouTubeDNN user-to-item recall results
- `itemcf_recall_dict.pkl` — ItemCF recall results
- `usercf_u2u2i_recall.pkl` — UserCF-based recall results
- `youtubednn_usercf_recall.pkl` — hybrid recall results

## Evaluation

Offline recall evaluation is implemented in `share.py`.
It typically uses the user's last click as the ground truth and reports Hit Rate / Recall-style metrics at different `topk` values.

Example output:

```text
Recall@10: 0.8056
NDCG@10: 0.4954
```

## Tips

- Keep the input click sequence sorted by timestamp.
- Do not lose interaction order when building `user_item_dict`.
- When using sequence models, padding tokens should be masked properly.
- If you change the working directory, update `data_path` and `save_path` accordingly.

## License

No license file is currently included. Add one if you plan to publish or share the project publicly.

