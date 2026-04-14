# News Recommendation Recall Pipeline

This repository implements a **multi-channel news recommendation / recall system** based on click logs.

The project focuses on multi-channel recall and ranking for news/article recommendation.

## Repository Layout

| Path | Description |
| --- | --- |
| `learn.py` | Main recall pipeline: builds item-based, user-based, embedding-based, and hybrid recall results. |
| `rank.py` | Ranking stage with LightGBM ranker/classifier and submission generation. |
| `share.py` | Shared utilities for data loading, feature engineering, and offline recall evaluation. |
| `i2i_sim.py` | Item-to-item similarity and ItemCF recall. |
| `u2u_sim.py` | User-to-user similarity and UserCF recall. |
| `YoutubeDnn.py` | YouTubeDNN-style two-tower recall model. |

## Data Files

Place the following files in the project root:

- `train_click_log.csv`
- `testA_click_log.csv`
- `articles.csv`
- `articles_emb.csv`

### Main Columns

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
- embedding columns such as `emb0`, `emb1`, ...

## Main Workflow

1. Load click logs and article metadata.
2. Build user click history and article interaction history.
3. Compute multiple recall channels:
   - item-based collaborative filtering (`ItemCF`)
   - user-based collaborative filtering (`UserCF`)
   - content embedding similarity
   - YouTubeDNN user/item embeddings
4. Merge recall results for ranking or ensemble.
5. Train ranking models in `rank.py` and export submission files.

## How to Run

Run scripts from the project root so relative paths resolve correctly:

```bash
python learn.py
```

To run ranking:

```bash
python rank.py
```

## Dependencies

Recommended packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`
- `faiss-cpu`
- `torch`
- `torch-rechub`
- `lightgbm`

Install them with:

```bash
pip install pandas numpy scikit-learn tqdm faiss-cpu torch torch-rechub lightgbm
```

## Outputs

The pipeline may generate files such as:

- `emb_i2i_sim.pkl`
- `itemcf_i2i_sim.pkl`
- `usercf_u2u_sim.pkl`
- `item_youtube_emb.pkl`
- `user_youtube_emb.pkl`
- `youtube_u2i_dict.pkl`
- `itemcf_recall_dict.pkl`
- `usercf_u2u2i_recall.pkl`
- `youtubednn_usercf_recall.pkl`
- `sample_submit.csv`


## Notes

- `share.py` uses the current working directory as both `data_path` and `save_path`, so run commands from the project root.
- `UserCF` can be memory-intensive on large datasets.
- Keep click histories sorted by timestamp.
- Do not lose interaction order when building user sequences.
- For sequence models, make sure padding tokens are masked correctly.

## License

No license file is currently included. Add one before publishing the project publicly.

