import dspy
from dspy.dsp.utils import deduplicate, normalize_text
import re
from nltk.tokenize import sent_tokenize


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

class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context = dspy.InputField(desc="may contain relevant facts")
    text = dspy.InputField(desc="between 1 to 2 sentences")
    faithfulness = dspy.OutputField(desc="boolean indicating if text is faithful to context")

def answer_correctness(example, pred, trace=None):
    assert hasattr(example, 'answer'), "Example does not have 'answer'."
    normalized_context = normalize_text(pred.paragraph)
    if isinstance(example.answer, str):
        gold_answers = [example.answer]
    elif isinstance(example.answer, list):
        gold_answers = example.answer
    else:
        raise ValueError("'example.answer' is not string or list.")
    return 1 if any(normalize_text(answer) in normalized_context for answer in gold_answers) else 0



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



def citation_faithfulness(pred, **kwargs):
    paragraph, context = pred.paragraph, kwargs['context']
    citation_dict = extract_text_by_citation(paragraph)

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
            assertion_msg.append(f"Make sure your output is based on the following context: '{context}'.")
        
        return False, " \n".join(assertion_msg), score
    return True, None, score


def assert_final(example, pred):
    if answer_correctness(example, pred):
        return 5
    else:
        return 0
        