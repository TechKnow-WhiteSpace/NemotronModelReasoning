# 🚀 Project Nemotron-RCCO: Curing Helpfulness Bias via "Structured Verbosity"

[![Award Category](https://img.shields.io/badge/NVIDIA_Open_Contribution_Award-Best_Data/Synthetic_Data_Method-76B900?style=for-the-badge)](https://www.kaggle.com/)
[![Model](https://img.shields.io/badge/Model-Nemotron_30B-blue?style=for-the-badge)]()
[![Methodology](https://img.shields.io/badge/Framework-RCCO-orange?style=for-the-badge)]()

**An end-to-end, procedurally verified Synthetic Data Factory designed to train Large Language Models how to *think*, rather than just how to talk.**

---

## 🧠 The Mission: Solving the Compute-Optimal Squeeze

During our research for the Kaggle Nemotron competition, we identified two massive, intersecting roadblocks limiting the reasoning capabilities of frontier instruction-tuned models:

1. **The "Helpfulness Bias":** Standard models (like Gemini 2.5 Flash and Claude 3.5 Sonnet) possess near-zero metacognitive inhibition. When faced with physically impossible scenarios or missing variables, their RLHF alignment forces them to blindly hallucinate solutions (scoring ~0% on our Impossible Physics Stress Test) rather than halting execution.
2. **The "Compute Squeeze" (vLLM Limits):** Pure reasoning models (like DeepSeek R1) solve this bias by utilizing massive `<think>` chains. However, these 2,000+ token internal monologues inevitably trigger hard memory and timeout limits on standard Kaggle inference infrastructure.

### 💡 The Solution: The RCCO Framework
We engineered the **Role, Context, Constraints, Output (RCCO)** prompting framework. Rather than compressing Chain-of-Thought (CoT) data into mindless brevity (which starves the model of test-time compute), our pipeline generates **"Structured Verbosity."**

We procedurally generate mathematically verified puzzles, force an elite Teacher model to solve them, and then use the RCCO framework to distill the reasoning. We strip all conversational filler but *rigorously retain every mathematical operation, state change, and logical branch*. 

The result? A highly dense, 111-token pseudo-code format that easily clears Kaggle's vLLM limits while giving the Nemotron 30B model the exact test-time compute it needs to excel.

---

## 📂 Repository Architecture

This repository contains the complete engineering pipeline required to generate, grade, and distill this high-fidelity dataset.

```text
NEMOTRONMODELREASONING/
│
├── data/                       # The Data Vault
│   ├── seed_files/             # Procedurally generated Python puzzles (The Prompts)
│   ├── synthetic_factory/      # Raw CoT output from the Teacher API (Pre-Grading)
│   └── gold_standard/          # The final, rejection-sampled, distilled training data
│
├── scripts/
│   ├── generation/             # 🧬 The Seed & Teacher Pipeline
│   │   ├── knapsack_generator.py
│   │   ├── logic_grid_generator.py
│   │   ├── stack_machine_generator.py
│   │   └── pipeline_generator.py
│   │
│   └── processing/             # ⚖️ The Deterministic Dual-State Judge
│       └── rejection_sampler.py
│
├── utility/                    # 🛠️ Data Distillation & Formatting
│   ├── make_concise_openai.py  # The RCCO "Structured Verbosity" compressor
│   └── data_consolidation.py
│
└── README.md