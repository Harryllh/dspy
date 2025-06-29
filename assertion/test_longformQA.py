import dspy
import dspy.predict
# from dspy.predict.predict_assert import PredictAssert
from typing import Literal
import re
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from dspy.datasets import HotPotQA
from typing import Callable, List, Tuple, Any
from dspy.adapters.chat_adapter import ChatAdapter
import os
from openai import OpenAI
from dspy.dsp.utils import deduplicate
import dspy
from dspy.clients.lm_local_arbor import ArborProvider
from assertion_chain import AssertionChain
from checkpoint import initialize_grpo, run_grpo_step, checkpoint, terminate_grpo


def correct_citation_format(paragraph):
    modified_sentences = []
    sentences = sent_tokenize(paragraph)
    for sentence in sentences:
        modified_sentences.append(sentence)
    citation_regex = re.compile(r'\[\d+\]\.')
    i = 0
    if len(modified_sentences) == 1:
      has_citation = bool(citation_regex.search(modified_sentences[i]))
    while i < len(modified_sentences):
      if len(modified_sentences[i:i+2]) == 2:
        sentence_group = " ".join(modified_sentences[i:i+2])
        has_citation = bool(citation_regex.search(sentence_group))
        if not has_citation:
            return False
        i += 2 if has_citation and i+1 < len(modified_sentences) and citation_regex.search(modified_sentences[i+1]) else 1
      else:
        return True
    return True

def has_citations(paragraph):
    return bool(re.search(r'\[\d+\]\.', paragraph))

def assert_citations(pred, **kwargs):
    paragraph = pred.paragraph
    if has_citations(paragraph) and correct_citation_format(paragraph):
        return True, None, 5
    else:
        # error_message = "Every 1-2 sentences should have citations: 'text... [x].'"
        error_message = "Make sure every 1-2 sentences has citations. If any 1-2 sentences lack citations, add them in 'text... [x].' format."
        return False, error_message, 0


def extract_text_by_citation(paragraph):
    citation_regex = re.compile(r'(.*?)(\[\d+\]\.)', re.DOTALL)
    parts_with_citation = citation_regex.findall(paragraph)
    citation_dict = {}
    for part, citation in parts_with_citation:
        part = part.strip()
        citation_num = re.search(r'\[(\d+)\]\.', citation).group(1)
        citation_dict.setdefault(str(int(citation_num) - 1), []).append(part)
    return citation_dict


class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context = dspy.InputField(desc="may contain relevant facts")
    text = dspy.InputField(desc="between 1 to 2 sentences")
    faithfulness = dspy.OutputField(desc="boolean indicating if text is faithful to context")

def citation_faithfulness(pred, **kwargs):
    paragraph, context = pred.paragraph, kwargs['context']
    # import pdb; pdb.set_trace()
    citation_dict = extract_text_by_citation(paragraph)
    # print(paragraph)
    # print(citation_dict)

    if not citation_dict:
        return False, None, 0
    context_dict = {str(i): context[i].split(' | ')[1] for i in range(len(context))}
    faithfulness_results = []
    unfaithful_citations = []
    check_citation_faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    for citation_num, texts in citation_dict.items():
        if citation_num not in context_dict:
            continue
        current_context = context_dict[citation_num]
        for text in texts:
            try:
                result = check_citation_faithfulness(context=current_context, text=text)
                is_faithful = result.faithfulness.lower() == 'true'
                faithfulness_results.append(is_faithful)
                if not is_faithful:
                    unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'context': current_context})
            except ValueError as e:
                faithfulness_results.append(False)
                unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'error': str(e)})
    final_faithfulness = all(faithfulness_results)  #TODO: change this to a percentage, not all
    if not faithfulness_results:
        return False, None, 0
    
    
    return final_faithfulness, unfaithful_citations, 5 * sum(faithfulness_results) / len(faithfulness_results)


def assert_faithful(pred, **kwargs):
    assertion_msg = []
    final_faithfulness, unfaithful_outputs, score = citation_faithfulness(pred, **kwargs)
    # print(final_faithfulness, unfaithful_outputs, score)
    # import pdb; pdb.set_trace()
    

    if unfaithful_outputs:
        unfaithful_pairs = [(output['text'], output['context']) for output in unfaithful_outputs]
        for _, context in unfaithful_pairs:
            if unfaithful_pairs == 0:
                assertion_msg.append(f"Make sure your output is based on the following context: '{context}'.")
    
    # score = 5 * len(unfaithful_outputs) / n
    if not assertion_msg:
        return True, None, score
    return False, " \n".join(assertion_msg), score



class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class GenerateCitedParagraph(dspy.Signature):
    """Generate a paragraph with citations."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    paragraph = dspy.OutputField(desc="includes citations")


class LongFormQAWithAssertions(dspy.Module): 
    def __init__(self, passages_per_hop=3, max_hops=2):
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        # self.generate_query = dspy.ChainOfThought("context: list[str], question: str -> query: str")
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        # self.generate_cited_paragraph = dspy.ChainOfThought("context: list[str], question: str -> paragraph: str")
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.assertion = AssertionChain(self.generate_cited_paragraph, max_retries=8)
        self.assertion.add_assertion(assert_citations)
        self.assertion.add_assertion(assert_faithful)
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
            pred = self.assertion(context=context, question=question)
            best_pred, data = pred
            print(best_pred)

        return data





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

current_model = local_lm_name
initialize_response = initialize_grpo(model=current_model)


prog = LongFormQAWithAssertions()
for i in range(len(trainset)):
    example = trainset[i]
    batch = prog(question=example.question)
    # print(a)
    step_response = run_grpo_step(model_name=current_model, batch=batch)
    current_model = step_response.json()["current_model"]

    if i == 10:
        checkpoint_response = checkpoint(checkpoint_name=f"checkpoint_{i}")
        last_checkpoint_name = checkpoint_response.json()["last_checkpoint"]

    if i == 20:
        break
        
    break

terminate_response = terminate_grpo()

# reference: https://github.com/stanfordnlp/dspy/blob/99d84558cb527880cb21c748f5f27172a0aa8169/examples/longformqa/longformqa_assertions.ipynb
