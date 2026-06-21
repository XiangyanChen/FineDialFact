import json
import numpy as np
import re
import string
import spacy
import nltk
from rank_bm25 import BM25Okapi
import os
from nltk.tokenize import sent_tokenize
from utils.prompt_wrapper import default_atomic_facts_prompt
from models.LLM import LLM
nltk.download("punkt")

class AtomicFactGenerator(object):
    def __init__(self, model_name, demon_dir, llm = None):
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = False
        self.demon_path = os.path.join(demon_dir, "demons.json" if self.is_bio else "demons_complex.json")
        if (llm is None):
            self.lm = LLM(model_name)
        else:
            self.lm = llm

        # get the demos
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def run(self, generations, cost_estimate=None):
        """Convert the generation into a set of atomic facts. Return a total words cost if cost_estimate != None."""
        assert isinstance(generations, list), "generation must be a list"
        paragraphs_list = list()
        for generation in generations:
            paragraphs_list.append([para.strip() for para in generation.split("\n") if len(para.strip()) > 0])
        return self.get_atomic_facts_from_paragraph(paragraphs_list, cost_estimate=cost_estimate)

    def get_atomic_facts_from_paragraph(self, paragraphs_list, cost_estimate=None):
        """
        :param paragraphs: list of list
        :param cost_estimate:
        :return:
        """
        sentences_list = []
        # para_breaks = []
        for paragraphs in paragraphs_list:
            sentences = list()
            for para_idx, paragraph in enumerate(paragraphs):
                # if para_idx > 0 :
                #     para_breaks.append(len(sentences))

                initials = detect_initials(paragraph)

                curr_sentences = sent_tokenize(paragraph)
                curr_sentences_2 = sent_tokenize(paragraph)

                curr_sentences = fix_sentence_splitter(curr_sentences, initials)
                curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

                # checking this, just to ensure the crediability of the sentence splitter fixing algorithm
                assert curr_sentences == curr_sentences_2, (paragraph, curr_sentences, curr_sentences_2)

                sentences += curr_sentences
            sentences_list.append(sentences)

        atoms_or_estimate = self.get_init_atomic_facts_from_sentence([sent for sentences in sentences_list for sent in sentences], cost_estimate=cost_estimate)

        if cost_estimate:
            return atoms_or_estimate
        else:
            atoms = atoms_or_estimate
        atomic_facts_pairs_list = []
        for sentences in sentences_list:
            atomic_facts_pairs = []
            for i, sent in enumerate(sentences):
                atomic_facts_pairs.append((sent, atoms[sent]))
            atomic_facts_pairs_list.append(atomic_facts_pairs)

        return atomic_facts_pairs_list


    def get_init_atomic_facts_from_sentence(self, sentences, cost_estimate=None):
        """Get the initial atomic facts from the sentences. Return a total words cost if cost_estimate != None."""

        demons = self.demons

        k = 1
        n = 1

        prompts = []
        prompt_to_sent = {}
        atoms = {}
        for sentence in sentences:
            if sentence in atoms:
                continue
            top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            prompt = ""

            for i in range(n):
                prompt = prompt + "Example: \n Input: {}\n Output:".format(
                    list(demons.keys())[i])
                for fact in demons[list(demons.keys())[i]]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"

            for match in top_machings:
                prompt = prompt + "Example: \n Input: {}\n Output:".format(match)
                for fact in demons[match]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"
            prompt = default_atomic_facts_prompt%(prompt, sentence)
            prompts.append(prompt)
            prompt_to_sent[prompt] = sentence

        if cost_estimate:
            total_words_estimate = 0
            for prompt in prompts:
                # if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
                #     continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            outputs = self.lm.generate(prompts, max_new_tokens=256)
            for output, prompt in zip(outputs, prompts):
                atoms[prompt_to_sent[prompt]] = text_to_sentences(output)

            return atoms


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_machings


def valid_hint(text):
    hints = ["Here is the",
             "Here are the",
             "Note that",
             "(Note:",
             "Note:",
             "Let me know if you have",
             "Since the input ",
             "The input",
             "(The input"]
    for hint in hints:
        if (text.startswith(hint)):
            return True
    return False

def filter_useless_token(text):
    useless_tokens = ["*", "+", "-"]
    for token in useless_tokens:
        if (text.startswith(token)):
            text = text[len(token):]
    return text

# transform InstructGPT output into sentences
def text_to_sentences(text):
    text = text.strip()
    # sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    sentences = text.split("\n")
    if len(sentences) > 0: 
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.' 
    else:
        sentences = []
    filtered_sentences = []

    for sentence in sentences:
        if (valid_hint(sentence) == True or len(sentence.strip()) == 0):
            continue
        sentence = filter_useless_token(sentence)
        sentence = sentence.strip()
        if (len(sentence) <= 1):
            continue
        filtered_sentences.append(sentence)
    return filtered_sentences


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False

def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)


    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)
        
    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities

def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):

    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split())==1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        # print (entities)
        new_facts = []
        for i, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any([fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for j, other_fact in enumerate(facts) if j != i]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities==covered_entities
        except Exception:
            new_facts = facts # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences


def main():
    test_list = ["Australian",
                 "The length of Frejus Road Tunnel is.",
                 "The clergy educated at Jesus College, Oxford include:",
                 "The Philadelphia area is home to many companies, including:",
                 "Kim Ki-duk",
                 "Japan, Canada, Russia, and the United States.",
                 "Casey Townsend wears number 23.",
                 "As the World Turns was released on October 1, 1957",
                 "He beat the American athlete, Bode Miller.",
                 "Bolvar is the capital of Bolva Province, Ecuador.",
                 "1923 to 1929",
                 "He has acted in over 40 films and television shows.",
                 "Berlin",
                 "Anderson House is located on Georgia State Route 78",
                 "Yes, it took place in Cupertino, California."]
    generator = AtomicFactGenerator("Meta-Llama-3-8B-Instruct", "factscore/demos")
    atomic_facts = generator.run(test_list)
    # atomic_facts, para_breaks = generator.run("Bridget Moynahan is an American actress, model and producer. She is best known for her roles in Grey’s Anatomy, I, Robot and Blue Bloods. She studied acting at the American Academy of Dramatic Arts, and …")
    # atomic_facts, para_breaks = generator.run("Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. He has been named Arsenal F.C's greatest ever player.\n\nHenry made his professional debut with Monaco in 1994 before signing for defending Serie A champions Juventus. However, limited playing time, coupled with disagreements with the club's hierarchy, led to him signing for Premier League club Arsenal for £11 million in 1999.")
    # print(text)
    for atom in atomic_facts:
        print(atom)
    # print(para_breaks)

if __name__ == "__main__":
    main()