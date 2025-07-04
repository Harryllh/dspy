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
        self.retry_prompts: List[str] = []
        self.original_sig = self.base_prog.predictors()[0].signature.instructions
        self.traces = []

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
    
    def update_reward(
        self,
        extra_reward: float
    ) -> None:
        """
        Add extra_reward to the "reward" field of each stored trace record.
        """
        for trace in self.traces:
            # ensure reward exists and is numeric
            trace['reward'] += extra_reward

    def get_trace(self):
        return self.traces
    
    def reset(self):
        self.base_prog.predictors()[0].signature.instructions = self.original_sig
        self.traces = []

    def __call__(self, **kwargs) -> Tuple[Any, List[int], int, List[Any], List[List[int]], List[int]]:
        best_pred = None
        best_scores: List[int] = []
        best_total = -1

        # for attempt in range(self.max_retries + 1):
            # if attempt == 0:
            #     chain = self.base_prog
        if self.retry_prompts:
            combined = " ".join(self.retry_prompts).strip()
            self.base_prog.predictors()[0].signature.instructions = self.original_sig + ' ' + combined  #Adding assertion msg as part of instructions
        
        # import pdb; pdb.set_trace()
        pred = self.base_prog(**kwargs)

        scores: List[int] = []
        retry_prompts = []
        for fn in self.assertions:
            with dspy.context(trace=[]):
                passed, prompt, score = fn(pred, **kwargs)
                # TODO; we don't need "passed"
                print(prompt, passed, score)
            scores.append(score)
            if not passed and prompt and prompt not in self.retry_prompts:
                self.retry_prompts.append(prompt)

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
        
        # self.traces['messages'] = inp_messages
        # self.traces['completion'].append({
        #         "role": all_messages[-1]["role"],
        #         "content": all_messages[-1]["content"],
        #     })
        # self.traces['reward'].append(float(total))


        self.traces.append({       #TODO: consider making this a list
            "messages": inp_messages,
            "completion": {
                "role": all_messages[-1]["role"],
                "content": all_messages[-1]["content"],
            },
            "reward": float(total),
        })

        # if total > best_total:
        #     best_total = total
        #     best_scores = scores
        #     best_pred = pred

        return pred
    























# from dataclasses import dataclass, field
# from typing import Any, List, Dict

# @dataclass
# class TraceNode:
#     name: str
#     inputs: Dict[str, Any]
#     output: Any
#     scores: List[int]
#     retry_prompts: List[str]
#     # child nodes: retries of this same module, then downstream modules
#     children: List["TraceNode"] = field(default_factory=list)

#     def to_dict(self):
#         return {
#             "name": self.name,
#             "inputs": self.inputs,
#             "output": str(self.output),
#             "scores": self.scores,
#             "retry_prompts": self.retry_prompts,
#             "children": [child.to_dict() for child in self.children],
#         }


# import json

# class AssertionChainWithTrace(AssertionChain):
#     def __call__(self, module_name: str, **kwargs) -> TraceNode:
#         """
#         Returns a TraceNode tree for this module.
#         `module_name` labels this node in the tree.
#         """
#         original_sig = self.base_prog.predictors()[0].signature.instructions

#         # root for this module
#         root = TraceNode(
#             name=module_name,
#             inputs=kwargs,
#             output=None,
#             scores=[],
#             retry_prompts=[],
#         )

#         best_total = -1
#         best_node = None

#         # for each attempt (first run + retries)
#         for attempt in range(self.max_retries + 1):
#             # reset instructions for retries
#             if attempt == 0:
#                 chain = self.base_prog
#             else:
#                 combined = " ".join(root.retry_prompts).strip()
#                 chain.predictors()[0].signature.instructions = original_sig + ' ' + combined

#             # run the chain and collect trace
#             with dspy.context(trace=[]) as run_trace:
#                 pred = chain(**kwargs)

#             # score assertions
#             scores, retry_prompts = [], []
#             for fn in self.assertions:
#                 passed, prompt, score = fn(pred, **kwargs)
#                 scores.append(score)
#                 if not passed and prompt:
#                     retry_prompts.append(prompt)

#             total = sum(scores)

#             # make a node for this attempt
#             attempt_node = TraceNode(
#                 name=f"{module_name}_attempt_{attempt}",
#                 inputs=kwargs,
#                 output=pred,
#                 scores=scores,
#                 retry_prompts=retry_prompts,
#             )
#             # attach the raw dspy trace as metadata if you want
#             attempt_node._dspy_trace = run_trace

#             root.children.append(attempt_node)

#             # track best
#             if total > best_total:
#                 best_total = total
#                 best_node = attempt_node

#             # prepare for next retry
#             root.retry_prompts = retry_prompts

#         # finally, record the best output
#         root.output = best_node.output
#         root.scores = best_node.scores
#         root.retry_prompts = best_node.retry_prompts

#         return root

# # Usage for a pipeline of modules:
# pipeline_traces = TraceNode(name="pipeline_root", inputs={}, output=None, scores=[], retry_prompts=[])
# # say you have modules: chain1, chain2, chain3
# trace1 = chain1("module1", input=foo)
# pipeline_traces.children.append(trace1)

# # feed trace1.output into next module
# trace2 = chain2("module2", data=trace1.output)
# pipeline_traces.children.append(trace2)

# # ...and so on

# # at end, dump to file:
# with open("full_trace.json", "w") as f:
#     json.dump(pipeline_traces.to_dict(), f, indent=2)
