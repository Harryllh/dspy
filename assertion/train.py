#!/usr/bin/env python3

import argparse
import json
from checkpoint import initialize_grpo, run_grpo_step, checkpoint, terminate_grpo
from longformQA import LongFormQAWithAssertions
from dspy.datasets import HotPotQA
from tqdm import tqdm
from utils import assert_final
from dspy.clients.lm_local_arbor import ArborProvider
import dspy
from dotenv import load_dotenv
import os
import time

load_dotenv()


# def style_check(batch):
def style_check(batch):
    """
    Validate that a batch is correctly formatted:
      - batch is a list
      - each item is a dict with keys: messages, completion, reward
      - messages is a list of {"role": ..., "content": ...}, and content is a string
      - completion and reward are lists of length exactly 1
    """
    # 1. Batch-level check
    assert isinstance(batch, list), "Batch must be a list"
    
    for idx, item in enumerate(batch):
        # 2. Item-level checks
        assert isinstance(item, dict), f"Item at index {idx} must be a dict"
        for key in ("messages", "completion", "reward"):
            assert key in item, f"Item at index {idx} missing '{key}' field"
        
        # 3. Messages check
        messages = item["messages"]
        assert isinstance(messages, list), f"'messages' in item {idx} must be a list"
        # for j, msg in enumerate(messages):
        #     assert isinstance(msg, dict), f"Message at index {j} in item {idx} must be a dict"
        #     assert "role" in msg, f"Message at index {j} in item {idx} missing 'role'"
        #     assert "content" in msg, f"Message at index {j} in item {idx} missing 'content'"
        #     assert isinstance(msg["content"], str), (
        #         f"'content' of message {j} in item {idx} must be a string"
        #     )
        
        # 4. Completion & reward check (length exactly 1)
        completion = item["completion"]
        reward = item["reward"]
        assert isinstance(completion, dict), f"'completion' in item {idx} must be a dict"
        assert isinstance(reward, float), f"'reward' in item {idx} must be a float"
    
    return True




def main(initial_model: str):

    dspy.settings.configure(rm=dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts'))

    dataset = HotPotQA(train_seed=1, train_size=300, eval_seed=2023, dev_size=300, test_size=0, keep_details=True)
    trainset = [x.with_inputs('question') for x in dataset.train]
    devset = [x.with_inputs('question') for x in dataset.dev]

    # current_model = local_lm_name
    # initialize_response = initialize_grpo(model=current_model)

    port = 7453
    current_model = initial_model

    prog = LongFormQAWithAssertions()

    initialize_response = initialize_grpo(model=current_model)

    tik = time.time()
    # with open("longformQAbatches_1.jsonl", "w", encoding="utf-8") as f:
    for i in tqdm(range(len(trainset))):
        print("current model: ", current_model)
        local_lm = dspy.LM(
            model=f"openai/arbor:{current_model}",
            provider=ArborProvider(),
            temperature=0.7,
            api_base=f"http://localhost:{port}/v1/",
            api_key="arbor",
            cache=False
        )

        dspy.configure(lm=local_lm)

        example = trainset[i]
        # for n in range(5):
        for retry in range(3):
            try:
                pred = prog(question=example.question)
                break
            except Exception as e:
                print(f"Error processing example {i}: {e}")

        if retry == 2:
            print(f"Failed to process example {i} after 3 retries.")
            continue
            
        assert hasattr(pred, 'reasoning') and hasattr(pred, 'paragraph'), "pred must have both .reasoning and .paragraph attributes"

        reward = assert_final(example, pred)
        prog.update_reward(reward)

        batches = prog.get_trace()

        for batch in batches:
            style_check(batch)

        for batch in batches:
            step_response = run_grpo_step(model_name=current_model, batch=batch)

            current_model = step_response.json()["current_model"]
            print(current_model)

        if i % 10 == 0 or i == len(trainset) - 1:
            checkpoint_response = checkpoint(checkpoint_name=f"checkpoint_{i}")
            last_checkpoint_name = checkpoint_response.json()["last_checkpoint"]
            print(f"[Checkpoint]: {last_checkpoint_name}")

        prog.reset()

        if i == 20:
            break
    
    tok = time.time()
    print(f"Time taken: {tok - tik} seconds")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a sequence of GRPO steps from a JSONL file."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Initial model name to use for GRPO (default: Qwen/Qwen3-8B)"
    )
    args = parser.parse_args()
    main(args.model)
