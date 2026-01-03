# URGENT: Network Error Fix

## Problem
HuggingFace dataset download failing with "Cannot assign requested address"

## Solution Applied

I've added:
1. **Retry logic** - 3 attempts with exponential backoff
2. **Synthetic data fallback** - Creates test stories if download fails

## What to Do on GPU Server

### Step 1: Pull Latest Code
```bash
cd AGI-experiments/
git pull
```

### Step 2: Run Training (Will Auto-Retry)
```bash
source venv/bin/activate

# Quick test - will use synthetic data if download fails
python train_symmetric.py --num_stories 100 --epochs 5 --batch_size 8
```

**What happens:**
- Tries to download TinyStories 3 times
- If all fail → automatically uses synthetic data
- Training continues regardless!

### Step 3: If Network Works Later

Once network is stable, delete cache and retry:
```bash
rm -rf data/tinystories_cache/
python train_symmetric.py --num_stories 1000 --epochs 10 --batch_size 32
```

## Alternative: Use Pre-downloaded Data

If you have TinyStories downloaded elsewhere:

```bash
# Copy to cache location
mkdir -p data/tinystories_raw/
cp /path/to/tinystories/*.parquet data/tinystories_raw/
```

## Synthetic Data Details

If network fails, creates simple stories like:
```
"Once upon a time there was a happy cat. The cat lived in a forest. 
One day the cat met a dog. They became good friends."
```

- 100-1000 stories
- Simple grammar
- Good for testing architecture
- Results won't be as good as real data, but training will work!

## To Verify It's Working

Training should show:
```
Attempt 1/3 to load dataset...
✗ Attempt 1 failed: Cannot assign requested address
Waiting 1 seconds before retry...
Attempt 2/3 to load dataset...
✗ Attempt 2 failed: Cannot assign requested address
Waiting 2 seconds before retry...
Attempt 3/3 to load dataset...
✗ All attempts failed. Using synthetic data...
Creating 100 synthetic stories...
✓ Dataset ready: 300 samples
```

Then training proceeds normally!

## Network Troubleshooting

Check if HuggingFace is reachable:
```bash
curl -I https://huggingface.co
ping huggingface.co
```

If blocked, might need proxy or VPN.

---

**Bottom line: Training will work even if network fails! Just pull the latest code and run.**
