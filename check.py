import json
for f in ["data/synthetic_factory/knapsack/knapsack_cot_dataset.jsonl", "data/synthetic_factory/stack_machine/stack_machine_cot_dataset.jsonl", "data/synthetic_factory/logic_grid/logic_grid_cot_dataset.jsonl"]:
    with open(f, 'r') as file:
        first = json.loads(file.readline())
        print(f"\n📄 File: {f}\n🔑 Keys: {first.keys()}\n🎯 Ground Truth: {json.dumps(first.get('ground_truth', {}), indent=2)}")