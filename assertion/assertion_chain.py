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
        max_retries: int = 6
    ):
        self.base_prog = base_prog
        self.assertions = assertions or []
        self.max_retries = max_retries
        self.adapter = ChatAdapter()
        self.retry_prompts: List[str] = []
        self.original_sig = self.base_prog.predictors()[0].signature
        self.original_inst = self.original_sig.instructions
        self.traces = []
        self.archived_traces = None
        self.total_score = 0
        self.is_last_module = is_last_module

    def add_assertion(
        self,
        assertion_fn,
        score
    ):
        """
        Add another assertion module.

        :param assertion_fn:   takes chain output, returns (passed, retry_prompt, score)
        """
        self.assertions.append(assertion_fn)
        self.total_score += score
    
    def update_reward(
        self,
        extra_reward: float
    ) -> None:
        """
        Add extra_reward to the "reward" field of each stored trace record.
        """
        if not self.traces:
            return

        all_rewards = [t['reward'] for t in self.traces]
        print("before:", all_rewards)

        if self.is_last_module:
            for t in self.traces:
                t['reward'] += extra_reward
        else:
            max_reward = max(all_rewards)
            # boost *all* traces tied for that highest reward
            for t in self.traces:
                if t['reward'] == max_reward:
                    t['reward'] += extra_reward
        
        print("after:", all_rewards)

    def get_trace(self):
        return self.traces
    
    def reset(self):
        self.original_sig.instructions = self.original_inst
        self.traces = []
        self.retry_prompts = []

    def __call__(self, **kwargs) -> Tuple[Any, List[int], int, List[Any], List[List[int]], List[int]]:
        best_pred = None
        best_total = -1

        inp_messages = self.adapter.format(
                                signature=self.original_sig,
                                inputs=kwargs,
                                demos=[] # TODO: Add support for demos
                            )
        

        for attempt in range(self.max_retries):
            if self.retry_prompts:
                combined = " ".join(self.retry_prompts).strip()
                self.original_sig.instructions = self.original_inst + ' ' + combined  #Adding assertion msg as part of instructions
            
            pred = self.base_prog(**kwargs)

            scores: List[int] = []
            retry_prompts = []
            for fn in self.assertions:
                with dspy.context(trace=[]):
                    passed, prompt, score = fn(pred, **kwargs)
                    # print("assertion msg: ", prompt)
                    # TODO; we don't need "passed"
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

            print(f"[attempt {attempt}] total score: {total} out of {self.total_score}")
            if total == self.total_score and attempt == 0:
                break
            
            trace = {
                'messages': inp_messages,
                'completion': {
                    "role": all_messages[-1]["role"],
                    "content": all_messages[-1]["content"],
                },
                'reward': float(total)

            }
            self.traces.append(trace)
            

        if total > best_total:
            best_total = total
            best_pred = pred

        return best_pred
    
