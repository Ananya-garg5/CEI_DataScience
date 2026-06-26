# CEI Data Science Assignments
**Student:** Ananya Garg

---

## Week 1 — Python & ML Fundamentals
**File:** `week1_Ananya_Garg.ipynb`

Covers Python basics, NumPy, Pandas, Linear Algebra, Statistics, and Probability.

---

## Week 2 — End-to-End ML Pipeline
**File:** `week2_Ananya_Garg.ipynb`  
**Dataset:** Tesla EA Deliveries & Production (2015–2025) — 2,640 rows × 12 columns
**Target:** `Estimated_Deliveries`, `Avg_Price_USD`

### Pipeline Structure
| Step | Description |
|------|-------------|
| Preprocessing | Null/duplicate check, dropped metadata column |
| EDA | Yearly trends, model/region breakdown, price distribution, correlation heatmap |
| Feature Engineering | Date ordinal, One-Hot Encoding for Region & Model |
| Regression Modeling | Ridge, Lasso, Random Forest, Gradient Boosting |
| Cross-Validation | 5-fold CV on best model |
| Hyperparameter Tuning | GridSearchCV on Gradient Boosting |
| Time Series Forecasting | Holt-Winters vs SARIMA — future 4Q forecast |
| Price Prediction | Separate GBM model predicting Avg_Price_USD |

---

## Week 3 — Customer Intelligence / Country Segmentation
**File:** `week3_Ananya_Garg.ipynb`
**Dataset:** Unsupervised Learning on Country Data — 167 countries × 9 features
**Goal:** Segment countries by socio-economic indicators and identify nations needing aid

### Pipeline Structure
| Step | Description |
|------|-------------|
| Cleaning | Strip whitespace, drop duplicates, force numeric types, median imputation |
| EDA | Correlation heatmap, boxplot per feature |
| Feature Scaling | StandardScaler on all numeric features |
| Elbow Method | K-Means inertia loop over k∈[2,10], elbow curve plotted |
| K-Means (k=3) | 3 clusters — Developed, Developing, Underdeveloped |
| DBSCAN | eps=1.5, min_samples=5 — secondary clustering method |
| PCA Visualization | 2D scatterplot of K-Means clusters |
| Cluster Profiling | Mean profile per cluster across all features |
| Key Insights | 5 observations on aid priority, economic zones, development gaps |

---

## Week 4 — CIFAR-10 Image Classification (ANN vs CNN)
**File:** `week4_Ananya_Garg.ipynb`
**Dataset:** CIFAR-10 — 60,000 images (32×32×3), 10 classes

### Pipeline Structure
| Step | Description |
|------|-------------|
| Data Loading | 50,000 train + 10,000 test images via `tf.keras.datasets` |
| Preprocessing | Pixel normalization 0–255 → 0–1, flattening for ANN |
| ANN | Dense (1024→512→256→128) with Dropout — test accuracy 38.81% |
| CNN | Conv2D (32→64→128) + BatchNorm + MaxPooling — test accuracy 69.79% |
| Training | Adam optimizer, sparse categorical crossentropy, 10 epochs |
| Learning Curves | Validation accuracy + loss comparison — ANN vs CNN |
| Augmented CNN | RandomFlip + RandomRotation + RandomZoom + EarlyStopping + ReduceLROnPlateau over 20 epochs — test accuracy 74.14% |
| Confusion Matrix | Side-by-side ANN vs CNN heatmap across all 10 classes |

---
## Week 5 — Text Generation (RNN vs LSTM vs GRU)
**File:** `week5_Ananya_Garg.ipynb`
**Corpus:** Custom AI/ML paragraph — 20 sentences, 145-word vocabulary

### Pipeline Structure
| Step | Description |
|------|-------------|
| Corpus & Cleaning | Custom 20-sentence AI/ML corpus, lowercased and stripped |
| Tokenization | Keras Tokenizer — 145 unique tokens, word-to-integer mapping |
| Sequence Preparation | N-gram sliding window → 180 sequences, pre-padded to length 12 |
| Vanilla RNN | Embedding(64) + SimpleRNN(128) + Dense(softmax) |
| LSTM | Embedding(64) + LSTM(128) + Dense(softmax) |
| GRU | Embedding(64) + GRU(128) + Dense(softmax) |
| Training | Adam optimizer, sparse categorical crossentropy, 200 epochs each |
| Loss Comparison | Line plot — RNN: 0.0098 / LSTM: 0.0690 / GRU: 0.0138 |
| Text Generation | Greedy decoding via np.argmax, 10 words per seed prompt |