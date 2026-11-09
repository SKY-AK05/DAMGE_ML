# Mask2Former Distillation Workflow & Azure VM Deployment Instructions

**Status:** APPROVED RUNBOOK  
**Branch:** `mask2former-distillation-migration`  

---

## 1. Overview of the Deployment Pipeline

```
  [ Local Machine (Windows) ]
       │
       ├─ 1. Audit & Architecture Migration (Complete)
       ├─ 2. All 52 Automated Tests Verified (Complete)
       ├─ 3. Review Git Diff & Stage Source Code Only
       └─ 4. Push Branch: mask2former-distillation-migration
              │
              ▼
       [ GitHub Remote ]
              │
              ▼
  [ Azure VM (Linux / GPU) ]
       │
       ├─ 5. git fetch & git checkout mask2former-distillation-migration
       ├─ 6. Validate real cache manifest against VM GPU environment
       ├─ 7. Execute 1-epoch integration smoke test on real GPU
       ├─ 8. Launch Prioritized Experiments:
       │      • E3: Single Teacher A
       │      • E4: Single Teacher B
       │      • E5: Fixed Multi-Teacher
       │      • E6: Adaptive Multi-Teacher
       └─ 9. Run automated hypothesis evaluation (PipelineEvaluator)
```

---

## 2. Local Machine Git Push Instructions

Run the following commands on the local machine inside the repository directory (`DAMGE_ML`):

```bash
# 1. Verify you are on the migration branch
git branch --show-current
# Expected: mask2former-distillation-migration

# 2. Check git status to ensure only code, tests, and documentation are modified/untracked
git status

# 3. Add modified and new files (excluding large assets, checkpoints, caches, or .env files)
git add pipelines/adaptive_distillation/
git add MASK2FORMER_MIGRATION_AUDIT.md
git add MASK2FORMER_ARCHITECTURE.md
git add MASK2FORMER_DISTILLATION_DESIGN.md
git add GPU_READINESS_REPORT.md
git add WORKFLOW_INSTRUCTIONS.md

# 4. Commit changes with a descriptive message
git commit -m "feat(distillation): complete pure Mask2Former multi-teacher distillation migration and test suite"

# 5. Push the migration branch to GitHub
git push -u origin mask2former-distillation-migration
```

---

## 3. Azure VM Execution Runbook

Once logged into the Azure GPU VM:

```bash
# 1. Navigate to project root
cd /home/orchvate/CAR_TRAIN1

# 2. Fetch and checkout the migration branch
git fetch origin
git checkout mask2former-distillation-migration
git pull origin mask2former-distillation-migration

# 3. Verify Python and CUDA environment
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 4. Run the 52 automated tests on the VM
python3 -m unittest discover -s pipelines/adaptive_distillation/tests -v

# 5. Validate Teacher Prediction Cache on the VM
python3 pipelines/adaptive_distillation/orchestrator.py --auto_run cache

# 6. Launch Prioritized Distillation Experiments:

# Experiment E3 — Single Teacher A:
python3 pipelines/adaptive_distillation/scripts/train_student.py \
  --config pipelines/adaptive_distillation/configs/e3_single_teacher_a.yaml

# Experiment E4 — Single Teacher B:
python3 pipelines/adaptive_distillation/scripts/train_student.py \
  --config pipelines/adaptive_distillation/configs/e4_single_teacher_b.yaml

# Experiment E5 — Fixed Multi-Teacher:
python3 pipelines/adaptive_distillation/scripts/train_student.py \
  --config pipelines/adaptive_distillation/configs/e5_fixed_multiteacher.yaml

# Experiment E6 — Adaptive Multi-Teacher:
python3 pipelines/adaptive_distillation/scripts/train_student.py \
  --config pipelines/adaptive_distillation/configs/e6_adaptive_multiteacher.yaml

# 7. Evaluate and Compare Experiments against Validation Split:
python3 pipelines/adaptive_distillation/scripts/evaluate_experiments.py \
  --config pipelines/adaptive_distillation/configs/e6_adaptive_multiteacher.yaml \
  --split val
```

