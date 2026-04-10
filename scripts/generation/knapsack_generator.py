import random
import json
import itertools

def generate_knapsack_puzzle(task_id):
    # We use 5-6 items. This is the "Goldilocks" zone for LLMs. 
    # Hard enough to require deep <think> planning, but small enough to fit in a context window.
    num_items = random.randint(5, 6)
    
    # Generate unique item names to make parsing predictable
    item_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    
    items = []
    total_weight = 0
    for i in range(num_items):
        weight = random.randint(2, 15)
        value = random.randint(10, 100)
        items.append({"name": item_names[i], "weight": weight, "value": value})
        total_weight += weight
        
    # Set the bag's capacity to roughly 40%-60% of the total weight
    # This guarantees the model MUST leave valuable items behind, forcing optimization
    capacity = int(total_weight * random.uniform(0.4, 0.6))
    
    # ==========================================
    # THE GROUND TRUTH SOLVER (Programmatic Answer Key)
    # ==========================================
    best_value = 0
    best_combo = []
    
    # We brute-force all possible combinations in Python to find the absolute maximum
    for r in range(1, num_items + 1):
        for combo in itertools.combinations(items, r):
            current_weight = sum(item["weight"] for item in combo)
            current_value = sum(item["value"] for item in combo)
            
            if current_weight <= capacity and current_value > best_value:
                best_value = current_value
                best_combo = [item["name"] for item in combo]
                
    # Sort the winning items alphabetically so the expected string is always in the same order
    best_combo.sort()
    combo_string = ", ".join(best_combo)
    
    # The exact string the model must output to pass Rejection Sampling
    expected_answer = f"Items: {combo_string} | Max Value: {best_value}"

    # Construct the Prompt for the Teacher Model
    items_text = "\n".join([f"- {item['name']}: Weight = {item['weight']}kg, Value = ${item['value']}" for item in items])
    
    prompt = (
        f"Task ID: {task_id}\n"
        f"You are an expert logistics planner. You have a knapsack that can hold a maximum weight of {capacity}kg.\n"
        f"You must choose a combination of the following items that maximizes the total value without exceeding the weight limit.\n\n"
        f"Available Items:\n{items_text}\n\n"
        f"Use <think> tags to calculate the weights and values of different combinations.\n"
        f"Format your final answer exactly like this: 'Items: [Item1, Item2] | Max Value: [Number]'.\n"
        f"Put your final formatted string inside \\boxed{{}}."
    )

    ground_truth = {
        "domain": "Combinatorial_Knapsack",
        "capacity": capacity,
        "max_value": best_value,
        "optimal_items": best_combo,
        "boxed_answer": expected_answer
    }

    return {"prompt": prompt, "ground_truth": ground_truth}

# Generate 1,000 unique optimization puzzles
print("Generating Combinatorial Optimization dataset...")
dataset = [generate_knapsack_puzzle(i) for i in range(1000)]

output_file = "../../data/seed_files/knapsack_dataset.jsonl"
with open(output_file, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

print(f"Successfully generated 1,000 tasks and saved to {output_file}.")