# make_sample.py
import json
import random
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Randomly sample examples from a JSONL file.")
    parser.add_argument("input", type=str, help="Path to the input JSONL file.")
    parser.add_argument("--n", type=int, default=200, help="Number of samples (default=200).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default=42).")
    args = parser.parse_args()

    input_path = args.input
    output_path = os.path.join(os.path.dirname(input_path), f"sampled_{args.n}.jsonl")

    with open(input_path, "r") as f:
        lines = f.readlines()

    data = [json.loads(line) for line in lines]

    random.seed(args.seed)
    sampled = random.sample(data, min(args.n, len(data)))

    with open(output_path, "w") as f:
        for ex in sampled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(sampled)} samples to {output_path}")

if __name__ == "__main__":
    main()

# python cot_eval/make_sample.py /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/aitz_test/all.jsonl --n 200