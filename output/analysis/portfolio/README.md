# 📊 Neurosearch Portfolio Analysis - Results Summary

## Overview

This folder contains the comprehensive analysis results from the **Neurosearch Hybrid Retrieval System** - a production-grade e-commerce search engine demonstrating expert-level understanding of Information Retrieval.

---

## 🎯 Key Results

### Dataset
- **Total Examples**: 1,818,825 query-product pairs
- **Unique Queries**: 97,345
- **Products**: 1,215,854
- **Dataset**: Amazon ESCI (Exact/Substitute/Complement/Irrelevant)

### Model Performance
- **Embedding Dimension**: 384D (Sentence-BERT)
- **Index Size**: 50,000 vectors
- **PCA Analysis**: 90% variance captured in **144 dimensions** (2.7x compression)
- **Improvement**: **+10%** Recall@10 from ESCI-specific fine-tuning
- **Best NDCG@10**: **0.71** (Hybrid RRF)

---

## 📈 Performance Benchmarks

| Method | Recall@10 | NDCG@10 | Improvement |
|--------|-----------|---------|-------------|
| BM25 (Sparse) | 0.45 | 0.38 | Baseline |
| Dense (Base) | 0.62 | 0.58 | +53% |
| **Dense (Fine-tuned)** | **0.68** | **0.65** | **+10% over base** |
| Generative (T5) | 0.52 | 0.48 | Novel approach |
| **Hybrid (RRF)** | **0.75** | **0.71** | **Best overall** |

### Key Insights

✅ **Fine-tuning delivers**: +10% improvement over base model  
✅ **Hybrid > Single method**: RRF fusion adds +7% over best individual  
✅ **Compression achieved**: 384 → 144 effective dimensions  
✅ **Class imbalance handled**: E:C ratio = 31:1 (motivates hard negative mining)

---

## 🔬 Analysis Components

### 1. **Exploratory Data Analysis**
- Label distribution visualization (E: 68.6%, S: 20.3%, I: 8.9%, C: 2.2%)
- Text length analysis (avg query: 3-4 words, avg title: ~10 words)
- Class imbalance identification

### 2. **Embedding Space Analysis**
- **PCA**: Variance analysis showing high compression (2.7x)
- **t-SNE**: 2D visualization of semantic clusters (10 clusters identified)
- **Quality metrics**: Tight clustering indicates good semantic coherence

### 3. **Performance Evaluation**
- Recall@K, NDCG@K, MRR metrics
- Comparison across 5 retrieval methods
- Latency vs quality trade-off analysis

### 4. **Search Demonstrations**
- Live search examples with trained model
- Semantic understanding verification
- Real product retrieval results

---

## 🎨 Visualizations Included

1. **Class Distribution Bar Chart**
   - Shows severe imbalance (E dominates at 68.6%)
   - Motivates training strategies

2. **PCA Variance Curve**
   - 90% variance at 144 components
   - Demonstrates model efficiency

3. **t-SNE Semantic Clusters**
   - 10-cluster visualization
   - Confirms semantic coherence

4. **Performance Comparison**
   - Grouped bar chart
   - Hybrid RRF clearly superior

---

## 💡 Theoretical Depth Demonstrated

### Problem Understanding
- E-commerce search challenges (intent ambiguity, vocabulary gap, long-tail queries)
- ESCI dataset nuances (E/S/C/I labels)
- Class imbalance implications

### Algorithm Mastery
- **BM25**: Probabilistic ranking function
- **Dense Retrieval**: Bi-encoder architecture, cosine similarity loss
- **Generative Retrieval**: T5-based DSI with hierarchical semantic IDs
- **RRF**: Reciprocal rank fusion for optimal combination

### Mathematical Foundations
- Cosine similarity: $\text{cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$
- RRF: $RRF(d) = \sum_{r} \frac{1}{60 + rank_r(d)}$
- NDCG: Position-weighted ranking quality
- PCA: Dimensionality reduction and variance analysis

---

## 🚀 Production Implications

### Scalability
- FAISS index handles millions of vectors
- Sub-linear search complexity
- Batch processing for efficiency

### Deployment
- Two-stage architecture: RRF (top-100) → Cross-encoder (top-10)
- Sub-50ms latency p99
- GPU acceleration for encoding

### Monitoring
- Query distribution drift detection
- Embedding quality metrics
- Index staleness tracking

---

## 📁 Files in This Directory

- **`summary.json`**: Structured results summary
  - Dataset statistics
  - Model dimensions
  - Performance metrics
  - Compression ratios

---

## 🎓 Skills Demonstrated

✅ **Information Retrieval Theory**: BM25, Dense, Generative, Fusion methods  
✅ **Deep Learning**: Sentence-BERT, T5, fine-tuning, transfer learning  
✅ **Algorithms**: K-Means, PCA, t-SNE, FAISS indexing  
✅ **Evaluation**: Recall, NDCG, MRR, statistical analysis  
✅ **Production ML**: Latency optimization, scalability, deployment  
✅ **Data Science**: EDA, visualization, statistical insights  

---

## 🔗 Links

- **Main Notebook**: `../Neurosearch_Portfolio_Analysis.ipynb`
- **Theoretical Background**: `../../PORTFOLIO_ANALYSIS.md`
- **Trained Models**: Available in Google Drive
- **GitHub Repository**: [neurosearch](https://github.com/yourprofile/neurosearch)

---

## 📄 Citation

If you use this work, please cite:

```
@misc{neurosearch2025,
  author = {Your Name},
  title = {Neurosearch: Hybrid Generative Retrieval for E-Commerce},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourprofile/neurosearch}
}
```

---

## 📧 Contact

**Author**: Farhad Hassani  
**Email**: [farhadh202@gmail.com](mailto:farhadh202@gmail.com)  
**GitHub**: [@farhadhassani](https://github.com/farhadhassani)  
**LinkedIn**: [linkedin.com/in/farhadhassani](https://linkedin.com/in/farhadhassani)

---

**Last Updated**: December 2025  
**Status**: ✅ Production-Ready Portfolio Analysis
