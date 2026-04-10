import random
import json

def generate_logic_grid(task_id):
    # The entities that will be randomly shuffled
    names = ["Alice", "Bob", "Charlie"]
    colors = ["Red", "Blue", "Green"]
    pets = ["Dog", "Cat", "Bird"]

    # Shuffle to ensure every generated puzzle is unique
    random.shuffle(names)
    random.shuffle(colors)
    random.shuffle(pets)

    # THE LOGIC TEMPLATE
    # We construct clues that require exactly 3 deductive hops to solve.
    # Hop 1: names[0] is paired with colors[0]
    clue1 = f"{names[0]} lives in the {colors[0]} house."
    
    # Hop 2: pets[1] is paired with colors[1]
    clue2 = f"The person with the {pets[1]} lives in the {colors[1]} house."
    
    # Hop 3: names[1] is paired with pets[2]
    clue3 = f"{names[1]} owns the {pets[2]}."

    """
    THE DEDUCTIVE PROOF (How the model must solve it):
    1. Who lives in colors[1]? It cannot be names[0] (they are in colors[0]). It must be names[1] or names[2].
    2. The person in colors[1] owns pets[1]. 
    3. We know names[1] owns pets[2]. Therefore, names[1] CANNOT live in colors[1].
    4. By elimination, names[2] MUST live in colors[1] and own pets[1].
    """

    target_person = names[2]
    correct_color = colors[1]
    correct_pet = pets[1]

    prompt = (
        f"Task ID: {task_id}\n"
        f"Three friends ({', '.join(sorted(names))}) each live in a differently colored house "
        f"({', '.join(sorted(colors))}) and own a different pet ({', '.join(sorted(pets))}).\n\n"
        f"Use the following clues to deduce the exact pairings:\n"
        f"1. {clue1}\n"
        f"2. {clue2}\n"
        f"3. {clue3}\n\n"
        f"Question: What is the exact combination for {target_person}? "
        f"Format your final answer as exactly: '[Name] lives in the [Color] house and owns the [Pet].' "
        f"Put your final formatted sentence inside \\boxed{{}}."
    )

    # The exact string our Rejection Sampler will look for to grade the Teacher Model
    expected_answer = f"{target_person} lives in the {correct_color} house and owns the {correct_pet}."

    ground_truth = {
        "domain": "Logic_Grid",
        "target_person": target_person,
        "correct_color": correct_color,
        "correct_pet": correct_pet,
        "boxed_answer": expected_answer
    }

    return {"prompt": prompt, "ground_truth": ground_truth}

# Generate 1,000 unique deductive logic puzzles
print("Generating Logic Grid Domain dataset...")
dataset = [generate_logic_grid(i) for i in range(1000)]

output_file = "../../data/seed_files/logic_grid_dataset.jsonl"
with open(output_file, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

print(f"Successfully generated 1,000 tasks and saved to {output_file}.")