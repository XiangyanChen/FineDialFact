
from few_shot.Examples import SemanticExampleRetriever
default_knowledge_prompt = "%sDialogue:\n%sGiven the above knowledge and dialogue, please respond to the input below and ensure the response is fluent and fact-consistent in English.\nInput:%s\nResponse:"
default_prompt = "Dialogue:\n%sGiven the above dialogue, please respond to the input below and ensure the response is fluent and fact-consistent in English.\nInput:%s\nResponse:"

# default_fact_checking_system_prompt = "The statement is a part of response to the dialogue. Evaluate the statement based on knowledge source and dialogue history. If it's not a factual claim (e.g., opinion or question), output 'non-support'. If it's a factual claim, check the knowledge source and dialogue history. Output 'true' if supported by direct evidence. Output 'false' if contradicted by direct evidence. Output 'non-support' if no direct evidence is found. Do not rely on external knowledge or assumptions."
default_fact_checking_system_prompt = "Instruction:\nThe statement is part of a response in a dialogue. Evaluate the statement strictly based on the provided knowledge source and dialogue history only.\n\nIf the statement is not a factual claim (e.g., opinion, question, or unclear assertion), output: \"not enough information.\"\n\nIf it is a factual claim:\n\nOutput true if the statement is directly supported by evidence in the knowledge source or dialogue history.\n\nOutput false if the statement is directly contradicted by the knowledge source or dialogue history.\n\nOutput not enough information if there is no direct evidence for or against the statement.\n\nImportant:\nDo not use your intern knowledge or make inferences.\n\nPlease only output your final answer and do not output any explanations."
default_fact_checking_wo_system_prompt = """
Evidence: %s\n
Dialogue history: %s\n
Statement: %s\n
Output:
"""

default_fact_checking_prompt = f"""
Evidence: %s\n
Dialogue history: %s\n
Statement: %s\n
{default_fact_checking_system_prompt}
"""
default_knowledge_select_string = "Please tell me which knowledge is important to answer the question of the dialogue.\n%s[SEP]Dialogue:\n%s[SEP]Question:%s\nImportant Knowledge:"
default_atomic_facts_prompt = "%sIf the following input is an incomplete sentence or a phrase, please output it exactly as it is. Otherwise, if it is a complete sentence, split it into atomic sentences based only on the given information, without adding any additional information or making inferences: \n Input: %s\n Output"

default_fact_checking_cot_system_prompt = "Instruction:\nThe statement is part of a response in a dialogue. Evaluate the statement strictly based on the provided knowledge source and dialogue history only.\n\nIf the statement is not a factual claim (e.g., opinion, question, or unclear assertion), output: \"not enough information.\"\n\nIf it is a factual claim:\n\nOutput true if the statement is directly supported by evidence in the knowledge source or dialogue history.\n\nOutput false if the statement is directly contradicted by the knowledge source or dialogue history.\n\nOutput not enough information if there is no direct evidence for or against the statement.\n\nImportant:\nDo not use your intern knowledge or make inferences.\n\nPlease think step by step and output your final answer."
default_fact_checking_cot_wo_system_prompt = """
Evidence: %s\n
Dialogue history: %s\n
Statement: %s\n
Output:
"""
default_fact_checking_cot_prompt = f"""
Evidence: %s\n
Dialogue history: %s\n
Statement: %s\n
{default_fact_checking_cot_system_prompt}
"""

retriever = SemanticExampleRetriever("few_shot/few_shot_examples.json")

def fact_checking_cot_input_wrapper_(examples, use_system_prompt=False):
    user_prompt_list = list()
    system_prompt_list = list()
    for dialogue, passages, last_utterance in zip(examples['dialogue'], examples['passages'], examples['last_utterance']):
        external_knowledge = ""
        for psg_idx, psg in enumerate(reversed(passages)):
            external_knowledge += "Title: %s\nText: %s\n\n"%(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
        external_knowledge = external_knowledge.strip()
        if use_system_prompt:
            user_prompt_list.append(default_fact_checking_cot_wo_system_prompt % (external_knowledge, dialogue, last_utterance))
            system_prompt_list.append(default_fact_checking_cot_system_prompt)
        else:
            user_prompt_list.append(default_fact_checking_cot_prompt % (external_knowledge, dialogue, last_utterance))
    return user_prompt_list, system_prompt_list

def fact_checking_few_shot_cot_input_wrapper_(examples, use_system_prompt=False):
    user_prompt_list = list()
    system_prompt_list = list()
    for dialogue, passages, last_utterance in zip(examples['dialogue'], examples['passages'], examples['last_utterance']):
        examples = retriever.search(last_utterance)
        example_text = "\n".join([f"Example {i}:\n{x}" for i, x in enumerate(examples)]) + "This is the end of examples.\n"
        external_knowledge = ""
        for psg_idx, psg in enumerate(reversed(passages)):
            external_knowledge += "Title: %s\nText: %s\n\n"%(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
        external_knowledge = external_knowledge.strip()
        if use_system_prompt:
            user_prompt_list.append(example_text + "\n" + default_fact_checking_cot_wo_system_prompt % (external_knowledge, dialogue, last_utterance))
            system_prompt_list.append(default_fact_checking_cot_system_prompt)
        else:
            user_prompt_list.append(example_text + "\n" + default_fact_checking_cot_prompt % (external_knowledge, dialogue, last_utterance))
    return user_prompt_list, system_prompt_list

def fact_checking_input_wrapper_(examples, use_system_prompt=False):
    user_prompt_list = list()
    system_prompt_list = list()
    for dialogue, passages, last_utterance in zip(examples['dialogue'], examples['passages'], examples['last_utterance']):
        external_knowledge = ""
        for psg_idx, psg in enumerate(reversed(passages)):
            external_knowledge += "Title: %s\nText: %s\n\n"%(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
        external_knowledge = external_knowledge.strip()
        if use_system_prompt:
            user_prompt_list.append(default_fact_checking_wo_system_prompt % (external_knowledge, dialogue, last_utterance))
            system_prompt_list.append(default_fact_checking_system_prompt)
        else:
            user_prompt_list.append(default_fact_checking_prompt % (external_knowledge, dialogue, last_utterance))
    return user_prompt_list, system_prompt_list

def baseline_input_wrapper(examples, max_turns):
    results = list()
    for question, history in zip(examples['A'], examples['history']):
        history = history
        context = extract_context(history, max_turns)
        prompt_data = default_prompt%(context, question)
        results.append(prompt_data)
    return results

def extract_context(history, max_turns):
    context = list()
    for i, text in enumerate(history):
        if (isinstance(text, str)):
            context.append("%s\n" % text)
        if (isinstance(text, dict)):
            if ("A" in text.keys()):
                context.append("A: %s\n" % text["A"])
            elif ("B" in text.keys()):
                context.append("B: %s\n" % text["B"])
        if (isinstance(text, list) or isinstance(text, tuple)):
            speaker = "Utterance"
            if text[0] == "question" or text[0] == "A":
                speaker = "A"
            if text[0] == "response" or text[0] == "B":
                speaker = "B"
            context.append("%s: %s\n" % (speaker, text[1]))
    return "".join(context[-max_turns:])
