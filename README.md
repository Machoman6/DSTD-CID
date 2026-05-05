# Dynamic Semantic Pooling and Temporal Dynamics Modeling for the Chinese CID Task - Reproduction Guide

## Quick Start for Reproduction

### 1. Environment Setup

```bash
# Create virtual environment with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10.12
- CUDA 12.8 (for GPU support)
- ~8GB GPU memory for BERT encoding

---

### 2. Data Preparation

The project expects event data in `cyberbullying_dataset/events/*.csv` format. Each CSV should contain:
- `content`: comment text
- `timestamp`: comment timestamp
- Event prior texts in `event_summary.csv` (name, content columns)

---

### 3. Reproduction Workflow

#### Step 0: Download Pre-trained Model

Download the Qwen3-Embedding model for text encoding:

```bash
python model_download.py
```

This downloads `Qwen/Qwen3-Embedding-0.6B` to the local cache.

---

#### Step 1: Build Dynamic Text Pools

```bash
cd dynamic_text_pool
python pipeline.py
```

**Input:** `cyberbullying_dataset/events/*.csv`  
**Output:** `hourly_outputs_0_0_1/<event_name>/all_pools.csv`

This processes each event and generates hourly aggregated text pools with relevance scores.

---

#### Step 2: Add Event Labels and Temporal Features

After generating the text pools, you need to add event labels and temporal history features:

**Add event labels (cyberbullying=1, non-cyberbullying=0):**
```bash
python add_event_labels.py
```

This script reads events from:
- `cyberbullying_dataset/cyberbullying/` (label=1)
- `cyberbullying_dataset/non_cyberbullying/` (label=0)

And adds the `event_label` column to each `all_pools.csv`.

**Add temporal history features:**
```bash
python batch_enhance_all_pools.py
```

This adds two columns:
- `hourly_total_history`: Cumulative comment count sequence per hour
- `hourly_malicious_history`: Cumulative malicious comment count sequence per hour

---

#### Step 3: Train ECTS Model

```bash
cd training
python train_early.py
```
