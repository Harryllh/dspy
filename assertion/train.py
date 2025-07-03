#!/usr/bin/env python3

import argparse
import json
from checkpoint import initialize_grpo, run_grpo_step, checkpoint, terminate_grpo

def main(filename: str, initial_model: str):
    # 1) initialize
    init_resp = initialize_grpo(model=initial_model)

    # 2) load all batches
    with open(filename, 'r', encoding='utf-8') as f:
        all_batches = [json.loads(line) for line in f]

    current_model = initial_model

    # 3) process each batch
    for i, batch in enumerate(all_batches):
        step_resp = run_grpo_step(model_name=current_model, batch=batch)
        print(f"[Batch {i}] step complete, response:", step_resp.json())
        current_model = step_resp.json().get("current_model", current_model)

        # checkpoint at batch 10
        if i == 10 or i == len(all_batches) - 1:
            cp_resp = checkpoint(checkpoint_name=f"checkpoint_{i}")
            last_cp = cp_resp.json().get("last_checkpoint")
            print(f"Checkpoint created: {last_cp}")


    # 4) terminate
    term_resp = terminate_grpo()
    print("Termination response:", term_resp.json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a sequence of GRPO steps from a JSONL file."
    )
    parser.add_argument(
        "--filename",
        help="Path to the input .jsonl file containing GRPO batches",
        default="longformQAbatches.jsonl"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Initial model name to use for GRPO (default: Qwen/Qwen3-8B)"
    )
    args = parser.parse_args()
    main(args.filename, args.model)
