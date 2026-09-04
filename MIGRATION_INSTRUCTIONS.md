# Migration Instructions: Moving from `master` to `main`

## Overview
This guide will help you merge all code from the `master` branch into `main` and set `main` as the primary branch going forward.

## Current State
- **master branch**: Contains all AGI experiments code (8 files)
- **main branch**: Only has LICENSE and README.md from initial repo creation
- **Goal**: Merge master into main, keep only main branch active

## Steps to Execute on Your Local Machine

### Step 1: Fetch Latest Changes
```bash
cd /path/to/AGI-experiments  # Navigate to your local repo
git fetch origin
```

### Step 2: Checkout the main branch
```bash
git checkout main
```

### Step 3: Merge master into main
```bash
git merge origin/master --allow-unrelated-histories -m "Merge master branch into main"
```

**Note**: The `--allow-unrelated-histories` flag is needed because master and main have different initial commits.

### Step 4: Resolve any conflicts (if they occur)
If there are conflicts in README.md:
- Decide which content to keep (likely the detailed README from master)
- Edit the file to resolve conflicts
- Run: `git add README.md`
- Run: `git commit -m "Resolve merge conflicts"`

### Step 5: Push the merged main branch
```bash
git push origin main
```

### Step 6: Update the default branch on GitHub (via web interface)
1. Go to: https://github.com/Cybernetic1/AGI-experiments/settings/branches
2. Under "Default branch", change from `main` (if not already) to ensure it's `main`
3. Confirm the change

### Step 7: (Optional) Delete the master branch
Once you've verified everything is on main:

**On GitHub (via web interface)**:
1. Go to: https://github.com/Cybernetic1/AGI-experiments/branches
2. Find the `master` branch
3. Click the trash icon to delete it

**Or via command line**:
```bash
git push origin --delete master
```

### Step 8: Clean up local branches
```bash
git branch -d master  # Delete local master branch
git remote prune origin  # Clean up remote tracking branches
```

## Verification

After migration, verify with:
```bash
git checkout main
git pull origin main
ls -la  # Should see all 8 files from master
```

Expected files on main:
- .gitignore
- README.md (detailed version from master)
- SCALING_TO_AGI.md
- STRUCTURE.md
- VARIABLES_VS_ENTITIES.md
- hierarchical_logic_network.py
- neural_logic_core.py
- requirements.txt
- LICENSE

## What If Something Goes Wrong?

If you need to abort the merge:
```bash
git merge --abort
```

If you pushed but want to undo:
```bash
git reset --hard origin/main~1  # Go back one commit
git push origin main --force  # Force push (use carefully!)
```

## Alternative: Simple Approach (Fresh Start on main)

If you prefer to avoid merge conflicts entirely:

```bash
# Checkout master and get all files
git checkout master
git pull origin master

# Create a new commit on main with all master files
git checkout main
git checkout master -- .  # Copy all files from master to main
git add .
git commit -m "Migrate all code from master to main"
git push origin main

# Then delete master as in Step 7 above
```

## Questions?
If you encounter any issues during migration, let me know and I can help troubleshoot!
