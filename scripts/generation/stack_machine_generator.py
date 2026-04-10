import random
import json

def generate_stack_execution(task_id):
    instructions = []
    stack = []
    
    # 8 to 12 instructions is the perfect length. 
    # Long enough to force deep CoT, short enough to avoid context limit degradation.
    num_ops = random.randint(8, 12)
    valid_ops = ["PUSH", "ADD", "MUL", "SUB", "SWAP"]

    # We programmatically build the instruction list AND solve it simultaneously
    for _ in range(num_ops):
        # Prevent errors by forcing a PUSH if the stack has fewer than 2 items
        if len(stack) < 2:
            op = "PUSH"
        else:
            op = random.choice(valid_ops)

        if op == "PUSH":
            val = random.randint(1, 15)
            instructions.append(f"PUSH {val}")
            stack.append(val)
            
        elif op == "ADD":
            a = stack.pop() # Top
            b = stack.pop() # Second
            res = a + b
            instructions.append("ADD")
            stack.append(res)
            
        elif op == "SUB":
            a = stack.pop() # Top
            b = stack.pop() # Second
            res = a - b     # Top minus Second
            instructions.append("SUB")
            stack.append(res)
            
        elif op == "MUL":
            a = stack.pop() # Top
            b = stack.pop() # Second
            res = a * b
            instructions.append("MUL")
            stack.append(res)
            
        elif op == "SWAP":
            a = stack.pop() # Top
            b = stack.pop() # Second
            instructions.append("SWAP")
            # Push them back in reverse order
            stack.append(a)
            stack.append(b)

    # The programmatic Ground Truth
    final_top = stack[-1]
    stack_str = ", ".join(map(str, stack))
    
    # The exact string the model must output to pass Rejection Sampling
    expected_answer = f"Final Top Value: {final_top} | Final Stack: [{stack_str}]"

    prompt = (
        f"Task ID: {task_id}\n"
        f"You are simulating a simple Stack Virtual Machine.\n"
        f"The machine supports the following operations:\n"
        f"- PUSH X: Push the integer X onto the top of the stack.\n"
        f"- ADD: Pop the top two elements, add them together, and push the result.\n"
        f"- SUB: Pop the top two elements (A is top, B is second), calculate (A - B), and push the result.\n"
        f"- MUL: Pop the top two elements, multiply them together, and push the result.\n"
        f"- SWAP: Pop the top two elements and push them back in reverse order.\n\n"
        f"Trace the state of the stack step-by-step using <think> tags. "
        f"Evaluate the following chronological sequence of instructions:\n"
        f"{', '.join(instructions)}\n\n"
        f"Format your final answer exactly as: 'Final Top Value: [Value] | Final Stack: [[Bottom, ..., Top]]'.\n"
        f"Put your final formatted string inside \\boxed{{}}."
    )

    ground_truth = {
        "domain": "Algorithmic_Execution",
        "instructions": instructions,
        "final_top": final_top,
        "final_stack": stack,
        "boxed_answer": expected_answer
    }

    return {"prompt": prompt, "ground_truth": ground_truth}

# Generate 1,000 unique algorithmic execution puzzles
print("Generating Algorithmic Execution dataset...")
dataset = [generate_stack_execution(i) for i in range(1000)]

output_file = "../../data/seed_files/stack_machine_dataset.jsonl"
with open(output_file, 'w') as f:
    for item in dataset:
        f.write(json.dumps(item) + '\n')

print(f"Successfully generated 1,000 tasks and saved to {output_file}.")