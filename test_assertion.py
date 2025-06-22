import dspy
import dspy.predict
from dspy.predict.predict_assert import PredictAssert
from typing import Literal
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from dspy.datasets import HotPotQA
from typing import Callable, List, Tuple, Any
import os
os.environ['OPENAI_API_KEY'] = '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'

# lm = dspy.LM('openai/gpt-4o')
lm = dspy.LM("gpt-4o-mini")
# lm = dspy.LM("gpt-3.5-turbo")
dspy.configure(lm=lm)


dspy.settings.configure(rm=dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts'))

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
        error_message = "Every 1-2 sentences should have citations: 'text... [x].'"
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
    citation_dict = extract_text_by_citation(paragraph)
    if not citation_dict:
        return False, None
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
    final_faithfulness = all(faithfulness_results)
    if not faithfulness_results:
        return False, None
    return final_faithfulness, unfaithful_citations


def assert_faithful(pred, **kwargs):
    assertion_msg = []
    _, unfaithful_outputs = citation_faithfulness(pred, **kwargs)
    if unfaithful_outputs:
        unfaithful_pairs = [(output['text'], output['context']) for output in unfaithful_outputs]
        for _, context in unfaithful_pairs:
            if unfaithful_pairs == 0:
                assertion_msg.append(f"Make sure your output is based on the following context: '{context}'.")
    
    if not assertion_msg:
        return True, None, 5
    return False, " \n".join(assertion_msg), 0

        
dataset = HotPotQA(train_seed=1, train_size=300, eval_seed=2023, dev_size=300, test_size=0, keep_details=True)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]



class CheckedChain:
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

    def add_assertion(
        self,
        assertion_fn,
    ) -> "CheckedChain":
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

        # New: full history
        all_preds: List[Any] = []
        all_scores_list: List[List[int]] = []
        all_totals: List[int] = []

        original_sig = self.base_prog.raw_signature
        retry_prompts: List[str] = []

        for attempt in range(self.max_retries + 1):
            if attempt == 0:
                chain = self.base_prog
            else:
                if not retry_prompts:
                    break
                combined = " ".join(retry_prompts).strip()
                retry_sig = dspy.Signature(original_sig, instructions=combined)
                chain = dspy.ChainOfThought(retry_sig)

            # run it
            pred = chain(**kwargs)

            # compute assertion scores & collect retry prompts
            scores: List[int] = []
            for fn in self.assertions:
                with dspy.context(trace=[]):
                    passed, prompt, score = fn(pred, **kwargs)
                scores.append(score)
                if not passed and prompt:
                    retry_prompts.append(prompt)

            total = sum(scores)

            # record in history
            all_preds.append(pred)
            all_scores_list.append(scores)
            all_totals.append(total)

            # update best
            if total > best_total:
                best_total = total
                best_scores = scores
                best_pred = pred

        # return best + full history
        return best_pred, all_preds, all_scores_list, all_totals


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
    def __init__(self, passages_per_hop=3):
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = dspy.ChainOfThought("context, question -> query")
        self.generate_cited_paragraph = dspy.ChainOfThought("context, question -> paragraph")

        self.assertion = CheckedChain(self.generate_cited_paragraph, max_retries=3)
        self.assertion.add_assertion(assert_citations)
        self.assertion.add_assertion(assert_faithful)

        super().__init__()
    
    def forward(self, question):
        context = []
        
        for hop in range(2):
            query = self.generate_query(context=context, question=question).query
            context += self.retrieve(query).passages
        
        with dspy.context(trace=[]):
            pred = self.assertion(context=context, question=question)
            best_pred, all_preds, all_scores_list, all_totals = dspy.settings.trace.copy()
            import pdb; pdb.set_trace()

        return best_pred



prog = LongFormQAWithAssertions()
for example in trainset:
    a = prog(question=example.question)
    print(a)
    break