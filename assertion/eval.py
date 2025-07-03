import dspy
from dspy.datasets import HotPotQA
from dspy.dsp.utils import deduplicate
from utils import GenerateSearchQuery, GenerateCitedParagraph
from utils import answer_correctness
from dspy.clients.lm_local_arbor import ArborProvider
from dspy.datasets import HotPotQA

class LongFormQA(dspy.Module): 
    def __init__(self, passages_per_hop=3, max_hops=2):
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops

        super().__init__()
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            # context += self.retrieve(query).passages
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        with dspy.context(trace=[]):
            pred = self.generate_cited_paragraph(context=context, question=question)

        return pred




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
local_lm_trained_name = "/scr-ssd/liheng/.arbor/storage/models/grpo:qwen3-8b:CstDN2:20250630_081151"
local_lm_trained = dspy.LM(
    model=f"openai/arbor:{local_lm_trained_name}",
    provider=ArborProvider(),
    temperature=0.7,
    api_base=f"http://localhost:{port}/v1/",
    api_key="arbor",
    cache=False
)

# dspy.configure(lm=local_lm)
dspy.configure(lm=local_lm_trained)
dspy.settings.configure(rm=dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts'))

dataset = HotPotQA(train_seed=1, train_size=300, eval_seed=2023, dev_size=300, test_size=0, keep_details=True)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

prog = LongFormQA()

evaluate = dspy.Evaluate(devset=devset, metric=answer_correctness, display_progress=True, num_threads=8)
evaluate(prog)





# for example in devset:
#     pred = prog(example)
#     right = answer_correctness(example, pred)
#     import pdb; pdb.set_trace()

