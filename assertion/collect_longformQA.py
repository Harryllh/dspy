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
        
        for hop in range(self.max_hops):
        # for hop in range(1):
            # query = self.generate_query(context=context, question=question).query
            query = self.generate_query_assertion[hop](context=context, question=question).query
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

port = 7453
# local_lm_name = "Qwen/Qwen2.5-7B"
# local_lm_name = "Qwen/Qwen3-8B"
local_lm_name = "/scr-ssd/liheng/.arbor/storage/models/grpo:qwen3-8b:MvXlSD:20250710_014257/checkpoints/checkpoint_896"
local_lm = dspy.LM(
    model=f"openai/arbor:{local_lm_name}",
    provider=ArborProvider(),
    temperature=0.7,
    api_base=f"http://localhost:{port}/v1/",
    api_key="arbor",
    cache=False
)

dspy.configure(lm=local_lm)
dspy.settings.configure(rm=dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts'))

dataset = HotPotQA(train_seed=1, train_size=300, eval_seed=2023, dev_size=300, test_size=0, keep_details=True)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

# current_model = local_lm_name
# initialize_response = initialize_grpo(model=current_model)


prog = LongFormQAWithAssertions()

with open("longformQAbatches_1.jsonl", "w", encoding="utf-8") as f:
    for i in tqdm(range(len(trainset))):
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
        reward = assert_final(example, pred)
        prog.update_reward(reward)
    
        batch = prog.get_trace()
        for item in batch:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        prog.reset()
    # print(a)







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
