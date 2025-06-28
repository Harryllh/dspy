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
        max_retries: int = 3
    ):
        self.base_prog = base_prog
        self.assertions = assertions or []
        self.max_retries = max_retries
        self.adapter = ChatAdapter()

    def add_assertion(
        self,
        assertion_fn,
    ):
        """
        Add another assertion module.

        :param assertion_fn:   takes chain output, returns (passed, retry_prompt, score)
        """
        self.assertions.append(assertion_fn)
        return self

    def __call__(self, **kwargs) -> Tuple[Any, List[int], int, List[Any], List[List[int]], List[int]]:
        best_pred = None
        best_scores: List[int] = []
        best_total = -1

        retry_prompts: List[str] = []

        data = []
        original_sig = self.base_prog.predictors()[0].signature.instructions

        for attempt in range(self.max_retries + 1):
            if attempt == 0:
                chain = self.base_prog
            else:
                if not retry_prompts:
                    break
                combined = " ".join(retry_prompts).strip()
                chain.predictors()[0].signature.instructions = original_sig + ' ' + combined  #Adding assertion msg as part of instructions
            
            pred = chain(**kwargs)

            scores: List[int] = []
            retry_prompts = []
            for fn in self.assertions:
                with dspy.context(trace=[]):
                    passed, prompt, score = fn(pred, **kwargs)
                    print(prompt, passed)
                scores.append(score)
                if not passed and prompt:
                    retry_prompts.append(prompt)

            total = sum(scores)

            inp_messages = self.adapter.format(
                                signature=self.base_prog.predictors()[0].signature,
                                inputs=kwargs,
                                demos=[] # TODO: Add support for demos
                            )
            all_messages = self.adapter.format_finetune_data(
                                signature=self.base_prog.predictors()[0].signature,
                                inputs=kwargs,
                                outputs=pred,
                                demos=[] # TODO: Add support for demos
                            )['messages']
            
            data.append({
                "messages": inp_messages,
                "completion": {
                    "role": all_messages[-1]["role"],
                    "content": all_messages[-1]["content"],
                },
                "reward": float(total),
            })

            if total > best_total:
                best_total = total
                best_scores = scores
                best_pred = pred

        return best_pred, data

