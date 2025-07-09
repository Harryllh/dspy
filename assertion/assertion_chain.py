import dspy
from typing import Literal
import re
from typing import Callable, List, Tuple, Any
from dspy.adapters.chat_adapter import ChatAdapter



class AssertionChain:
    """
    Wraps a dspy.ChainOfThought and a list of
    (assertion_fn, retry_prompt, score) tuples.
    On call, it:
      - runs up to max_retries+1 total attempts:
          • first with the original base_chain
          • then, for each retry, re-runs base_chain with only failed
            retry_prompts concatenated as instructions
      - for each run, scores each assertion (score or 0)
      - returns the Result and scores from the run with the highest total score
    """
    def __init__(
        self,
        base_prog: dspy.ChainOfThought,
        assertions: List[Tuple[Callable[[Any], bool], str, int]] = None,
        is_last_module = False,
        max_retries: int = 3
    ):
        self.base_prog = base_prog
        self.assertions = assertions or []
        self.max_retries = max_retries
        self.adapter = ChatAdapter()
        self.retry_prompts: List[str] = []
        self.original_sig = self.base_prog.predictors()[0].signature
        self.original_inst = self.original_sig.instructions
        self.traces = {
            "messages": "",
            "completion": [],
            "reward": []
        }
        self.archived_traces = None
        
        self.is_last_module = is_last_module

    def add_assertion(
        self,
        assertion_fn,
    ):
        """
        Add another assertion module.

        :param assertion_fn:   takes chain output, returns (passed, retry_prompt, score)
        """
        self.assertions.append(assertion_fn)
    
    def update_reward(
        self,
        extra_reward: float
    ) -> None:
        """
        Add extra_reward to the "reward" field of each stored trace record.
        """
        all_rewards = self.traces['reward']
        print("before:", all_rewards)
        if self.is_last_module:
            for i in range(len(all_rewards)):
                all_rewards[i] += extra_reward
        else:
            max_reward = max(all_rewards)
            # boost *all* traces tied for that highest reward
            for i in range(len(all_rewards)):
                if all_rewards[i] == max_reward:
                    all_rewards[i] += extra_reward
        
        print("after:", all_rewards)

    def get_trace(self):
        return self.traces
    
    def reset(self):
        self.original_sig.instructions = self.original_inst
        self.traces = {
            "messages": "",
            "completion": [],
            "reward": []
        }
        self.retry_prompts = []

    def __call__(self, **kwargs) -> Tuple[Any, List[int], int, List[Any], List[List[int]], List[int]]:
        best_pred = None
        best_total = -1

        inp_messages = self.adapter.format(
                                signature=self.original_sig,
                                inputs=kwargs,
                                demos=[] # TODO: Add support for demos
                            )
        self.traces['messages'] = inp_messages

        for attempt in range(self.max_retries + 1):
            # if attempt == 0:
            #     chain = self.base_prog
            if self.retry_prompts:
                combined = " ".join(self.retry_prompts).strip()
                self.original_sig.instructions = self.original_inst + ' ' + combined  #Adding assertion msg as part of instructions
            
            pred = self.base_prog(**kwargs)

            scores: List[int] = []
            retry_prompts = []
            for fn in self.assertions:
                with dspy.context(trace=[]):
                    passed, prompt, score = fn(pred, **kwargs)
                    # TODO; we don't need "passed"
                    print(prompt, passed, score)
                scores.append(score)
                if prompt and prompt not in self.retry_prompts:
                    self.retry_prompts.append(prompt)

            total = sum(scores)

            all_messages = self.adapter.format_finetune_data(
                                signature=self.original_sig,
                                inputs=kwargs,
                                outputs=pred,
                                demos=[] # TODO: Add support for demos
                            )['messages']
            
            self.traces['completion'].append({
                    "role": all_messages[-1]["role"],
                    "content": all_messages[-1]["content"],
                })
            self.traces['reward'].append(float(total))

            # check if this trace has reached the max possible score
            if total == len(self.assertions) * 5:
                break

        if total > best_total:
            best_total = total
            best_pred = pred

        return best_pred
    
