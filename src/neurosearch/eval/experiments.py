import pandas as pd
from .metrics import ndcg_at_k, recall_at_k, mrr_at_k


def run_experiment(model, dataset, top_k=10):
    """Run a simple evaluation experiment.
    `model` should have a `search(query, top_k)` method returning a list of doc IDs.
    `dataset` is a DataFrame with columns ['query', 'relevant_ids'] where relevant_ids is a list of true doc IDs.
    Returns a dict of metric scores.
    """
    ndcgs, recalls, mrrs = [], [], []
    for _, row in dataset.iterrows():
        query = row['query']
        true_ids = set(row['relevant_ids'])
        results = model.search(query, top_k=top_k)
        retrieved_ids = [doc_id for doc_id, _ in results]
        # Build relevance list (1 if retrieved doc is relevant, else 0)
        relevances = [1 if doc_id in true_ids else 0 for doc_id in retrieved_ids]
        ndcgs.append(ndcg_at_k(relevances, top_k))
        recalls.append(recall_at_k(relevances, top_k))
        mrrs.append(mrr_at_k(relevances, top_k))
    return {
        'ndcg@k': sum(ndcgs) / len(ndcgs) if ndcgs else 0,
        'recall@k': sum(recalls) / len(recalls) if recalls else 0,
        'mrr@k': sum(mrrs) / len(mrrs) if mrrs else 0,
    }
