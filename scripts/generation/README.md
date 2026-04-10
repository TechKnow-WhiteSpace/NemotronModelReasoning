# 🧬 Synthetic Data Generation: The Seed & Teacher Pipeline

## Overview
This directory contains the core engines of our Synthetic Data Factory. Generating high-quality reasoning data for Large Language Models (LLMs) is notoriously difficult due to the "Helpfulness Bias"—the tendency of models to hallucinate answers when faced with complex logic or missing variables. 

To overcome this, we rely on **Procedural Generation combined with Programmatic Ground Truth**. Instead of hoping a Teacher model gets the math right, we use Python to dynamically build logic puzzles, calculate the exact, indisputable answer, and *then* ask the Teacher model to explain its reasoning. 

This guarantees that every piece of Chain-of-Thought (CoT) data we generate is anchored to an absolute mathematical truth.

## 🛠️ The Procedural Generators

We developed three distinct domains of logic puzzles to ensure diverse test-time compute training for the Nemotron model.

### 1. `knapsack_generator.py` (Combinatorial Optimization)
* **The Logic:** Generates classic 0/1 Knapsack problems with 5-6 random items.
* **The Ground Truth:** Uses a brute-force `itertools` combination solver in Python to find the absolute maximum value. We purposefully set the bag capacity to 40%-60% of total weight to force the model to make difficult optimization choices.
* **Output:** Creates 1,000 unique optimization puzzles.

### 2. `logic_grid_generator.py` (Deductive Elimination)
* **The Logic:** Generates 3-hop deductive logic puzzles (e.g., "Alice lives in the Red house," "The person with the Dog lives in the Blue house").
* **The Ground Truth:** The script programmatically maps the entities (Names, Colors, Pets) and guarantees the deductive path is completely solvable by elimination.
* **Output:** Creates 1,000 unique deductive logic puzzles.

### 3. `stack_machine_generator.py` (Algorithmic Execution)
* **The Logic:** Simulates a virtual Stack Machine executing 8-12 chronological operations (`PUSH`, `ADD`, `SUB`, `MUL`, `SWAP`).
* **The Ground Truth:** The Python script acts as the compiler, evaluating the stack in real-time to generate the exact final Top Value and Stack Array.
* **Output:** Creates 1,000 unique algorithmic execution puzzles.

---

## 👨‍🏫 The Teacher Engine

### `pipeline_generator.py`
This script takes the procedural "Seed Data" generated above and passes it to our elite Teacher model (Gemini 2.5 Pro). 
* **RCCO Prompting:** It enforces our strict Role, Context, Constraints, Output framework. It mandates that Gemini must use `<think>...</think>` tags for step-by-step logic and strictly output the final answer inside `\boxed{}`.
* **Resiliency:** Features an exponential backoff wrapper (`tenacity`) to handle API rate limits and a state-tracking system to avoid re-running successful or blacklisted IDs.

### `format_kaggle_data.py`
A utility script that ingests raw CSV training data from Kaggle, wraps it in our required RCCO prompt structure (demanding `<think>` tags and a `\boxed{}` answer), and converts it into a JSONL seed file ready for the Teacher API.