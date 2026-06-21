from utils import prompt_wrapper
from models.LLM import LLM

class FactVerification:

    def __init__(self, model_name, train_output_root_dir=None):
        self.llm = LLM(model_name, train_ckpt_path=train_output_root_dir)
        self.use_system_prompt = True

    def verify(self, dialogue, last_utterance, evidence, mode, temperature):
        examples = {}
        examples['dialogue'] = dialogue
        examples['passages'] = evidence
        examples['last_utterance'] = last_utterance
        if (mode == "plain"):
            prompt_list, system_prompt = prompt_wrapper.fact_checking_input_wrapper_(examples, self.use_system_prompt)
        elif (mode == "cot"):
            prompt_list, system_prompt = prompt_wrapper.fact_checking_cot_input_wrapper_(examples, self.use_system_prompt)
        elif (mode == "few-shot-cot"):
            prompt_list, system_prompt = prompt_wrapper.fact_checking_few_shot_cot_input_wrapper_(examples,
                                                                                         self.use_system_prompt)
        else:
            raise NotImplementedError
        output_list = self.llm.generate(prompt_list, system_prompt=system_prompt, max_new_tokens=2048, temperature=temperature)
        support_list = list()
        explanation_list = list()
        for output in output_list:
            generated_answer = output.lower()
            explanation_list.append(generated_answer)
            if "true" in generated_answer or "false" in generated_answer or "not enough information" in generated_answer:
                # Using rfind instead of index to search from right to left
                try:
                    NEI_index = generated_answer.rfind("not enough information")
                except:
                    NEI_index = -1  # Not found

                try:
                    true_index = generated_answer.rfind("true")
                except:
                    true_index = -1  # Not found

                try:
                    false_index = generated_answer.rfind("false")
                except:
                    false_index = -1  # Not found

                # If keyword not found, set index to -1 (not found)
                if NEI_index == -1:
                    NEI_index = -1
                if true_index == -1:
                    true_index = -1
                if false_index == -1:
                    false_index = -1

                # Determine which keyword appears last (highest index)
                if true_index > NEI_index and true_index > false_index and true_index != -1:
                    is_supported = "SUPPORTS"
                elif false_index > NEI_index and false_index > true_index and false_index != -1:
                    is_supported = "REFUTES"
                elif NEI_index != -1:
                    is_supported = "NOT ENOUGH INFO"
                # Additional checks for single keyword scenarios
                elif "true" in generated_answer and "false" not in generated_answer and "not enough information" not in generated_answer:
                    is_supported = "SUPPORTS"
                elif "false" in generated_answer and "true" not in generated_answer and "not enough information" not in generated_answer:
                    is_supported = "REFUTES"
                elif "not enough information" in generated_answer and "true" not in generated_answer and "false" not in generated_answer:
                    is_supported = "NOT ENOUGH INFO"
                else:
                    is_supported = "NOT ENOUGH INFO"
            else:
                is_supported = "NOT ENOUGH INFO"
            support_list.append(is_supported)
        return support_list, explanation_list, prompt_list, system_prompt