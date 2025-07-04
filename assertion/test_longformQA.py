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
from openai import OpenAI
from dspy.dsp.utils import deduplicate, normalize_text
import dspy
from dspy.clients.lm_local_arbor import ArborProvider
from assertion_chain import AssertionChain
from checkpoint import initialize_grpo, run_grpo_step, checkpoint, terminate_grpo
import json
from utils import GenerateSearchQuery, GenerateCitedParagraph, assert_faithful, assert_citations
from utils import answer_correctness


def assert_final(example, pred):
    if answer_correctness(example, pred):
        return 5
    else:
        return 0




class LongFormQAWithAssertions(dspy.Module): 
    def __init__(self, passages_per_hop=3, max_hops=2):
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        # self.generate_query = dspy.ChainOfThought("context: list[str], question: str -> query: str")
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        # self.generate_cited_paragraph = dspy.ChainOfThought("context: list[str], question: str -> paragraph: str")
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        # TODO: generate_cited_paragraph_assertion
        self.generate_cited_paragraph_assertion = AssertionChain(self.generate_cited_paragraph, max_retries=8)
        self.generate_cited_paragraph_assertion.add_assertion(assert_citations)
        self.generate_cited_paragraph_assertion.add_assertion(assert_faithful)
        self.max_hops = max_hops

        super().__init__()
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            # print("query", query)
            # context += self.retrieve(query).passages
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        with dspy.context(trace=[]):
            pred = self.generate_cited_paragraph_assertion(context=context, question=question)

        return pred
    
    def update_reward(self, reward):
        self.generate_cited_paragraph_assertion.update_reward(reward)
    
    def get_trace(self):
        return self.generate_cited_paragraph_assertion.get_trace()
    
    def reset(self):
        self.generate_cited_paragraph_assertion.reset()


# TODO: make reward assignment configurable. add final reward. make this a tree and we can write to some documents for each module.

port = 7453
local_lm_name = "Qwen/Qwen3-8B"
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

with open("longformQAbatches.jsonl", "w", encoding="utf-8") as f:
    for i in tqdm(range(len(trainset))):
        example = trainset[i]
        for n in range(5):
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
        f.write(json.dumps(batch, ensure_ascii=False) + "\n")

        prog.reset()
        import pdb; pdb.set_trace()

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
