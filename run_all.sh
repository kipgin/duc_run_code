#!/bin/bash

# ==============================================================================
# 1. OPTIONAL: Install & Run in tmux (prevents disconnection on long runs)
# ==============================================================================
# Step 1: Install tmux
# sudo apt-get update && sudo apt-get install -y tmux
#
# Step 2: Start a named tmux session
# tmux new -s hw4
#
# Step 3: Useful tmux shortcuts:
# - Detach session:  Press 'Ctrl+B', release, then press 'D'
# - Reattach later:  tmux attach -t hw4
# - List sessions:   tmux ls
# - Kill session:    tmux kill-session -t hw4

# ==============================================================================
# 2. OPTIONAL: Set up Hugging Face CLI & Login
# ==============================================================================
# pip install -U "huggingface_hub[cli]"
# hf auth login
# # (or: huggingface-cli login)

# ==============================================================================
# 3. OPTIONAL: Download precomputed data from Hugging Face
# ==============================================================================
# huggingface-cli download kipgin/precomputed_vector --repo-type dataset --local-dir data/raw
# # (or using hf: hf download kipgin/precomputed_vector --repo-type dataset --local-dir data/raw)

# ==============================================================================
# 4. Install Dependencies & Cache Configuration
# ==============================================================================
# pip install torch fswlib pot scikit-learn scipy numpy matplotlib tqdm
# export WMD_CACHE_ROOT=/path/to/your/existing/cache

# ==============================================================================
# 5. Run HW4 & HW4b Pipeline
# ==============================================================================
# One dataset at a time (recommended — lets you watch progress bars):
# python run_hw4.py --dataset amazon
# python run_hw4.py --dataset classic
python run_hw4.py --dataset reuters

# Or all datasets:
# python run_hw4.py --dataset all

# Generate HW4 report
python make_report.py

# Run HW4b (FSW Shortlist + Exact WMD-W2 rerank)
python run_hw4b.py --dataset reuters

# Generate HW4b report
python make_report_4b.py