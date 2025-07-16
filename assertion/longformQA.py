import dspy
from typing import Literal
import re
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from tqdm import tqdm
from dspy.datasets import HotPotQA
from typing import Callable, List, Tuple, Any
from dspy.adapters.chat_adapter import ChatAdapter
import os
from dspy.dsp.utils import deduplicate, normalize_text
import dspy
from dspy.clients.lm_local_arbor import ArborProvider
from assertion_chain import AssertionChain
from checkpoint import initialize_grpo, run_grpo_step, checkpoint, terminate_grpo
import json
from utils import GenerateSearchQuery, GenerateCitedParagraph, assert_faithful, assert_citations, assert_query_length, assert_query_content, assert_final
from utils import answer_correctness



class LongFormQAWithAssertions(dspy.Module): 
    def __init__(self, passages_per_hop=3, max_hops=2):
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        # self.generate_query = dspy.ChainOfThought("context: list[str], question: str -> query: str")
        self.generate_query_raw = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]

        self.generate_query_assertion = [AssertionChain(generate_query, is_last_module=False) for generate_query in self.generate_query_raw]
        for query_assertion in self.generate_query_assertion:
            query_assertion.add_assertion(assert_query_content, 5)
            
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.generate_cited_paragraph_assertion = AssertionChain(self.generate_cited_paragraph, is_last_module=True)
        self.generate_cited_paragraph_assertion.add_assertion(assert_citations, 3)
        self.generate_cited_paragraph_assertion.add_assertion(assert_faithful, 5)  # Make sure the score matches what's defined in the assertion function
        self.max_hops = max_hops

        super().__init__()
    
    def forward(self, question):
        context = []
        queries = []

        for hop in range(self.max_hops):
        # for hop in range(1):
            # query = self.generate_query(context=context, question=question).query
            query = self.generate_query_assertion[hop](context=context, question=question, existing_queries=queries).query
            queries.append(query)
            # context += self.retrieve(query).passages
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        with dspy.context(trace=[]):
            pred = self.generate_cited_paragraph_assertion(context=context, question=question)

        return pred
    
    def update_reward(self, reward):
        for query_assertion in self.generate_query_assertion:
            query_assertion.update_reward(reward)
        self.generate_cited_paragraph_assertion.update_reward(reward)
    
    def get_trace(self):
        return [self.generate_cited_paragraph_assertion.get_trace()] + [query_assertion.get_trace() for query_assertion in self.generate_query_assertion]
    
    def reset(self):
        self.generate_cited_paragraph_assertion.reset()
        for query_assertion in self.generate_query_assertion:
            query_assertion.reset()


# TODO: make reward assignment configurable. add final reward. make this a tree and we can write to some documents for each module.








#     step_response = run_grpo_step(model_name=current_model, batch=batch)
#     print('step complete')
#     current_model = step_response.json()["current_model"]

#     if i == 10:
#         checkpoint_response = checkpoint(checkpoint_name=f"checkpoint_{i}")
#         last_checkpoint_name = checkpoint_response.json()["last_checkpoint"]

#     if i == 20:
#         break
        
#     # break

# terminate_response = terminate_grpo()

# reference: https://github.com/stanfordnlp/dspy/blob/99d84558cb527880cb21c748f5f27172a0aa8169/examples/longformqa/longformqa_assertions.ipynb
