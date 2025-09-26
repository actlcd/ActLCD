# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
# Ref: https://github.com/voidism/DoLa

import re
import os
import json
import random
import transformers
from tqdm import tqdm, trange
import argparse

import ssl
import urllib.request
import zipfile

from dola import DoLa

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 6
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"
SHORT_ANSWER_TRIGGER = "answer is" # for long answer

def load_jsonl(file_path, is_gzip=False):
    # Format of each line in StrategyQA:
    # {"qid": ..., "term": ..., "description": ..., "question": ..., "answer": ..., "facts": [...], "decomposition": [...]}
    
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        items = json.load(f)
        for item in items:
            new_item = dict(
                qid=item.get('qid', None),
                # term=item.get('term', None),
                # description=item.get('description', None),
                question=item.get('question', None),
                answer=item.get('answer', None),
                # facts=item.get('facts', []),
                # decomposition=item.get('decomposition', [])
            )
            list_data_dict.append(new_item)
    return list_data_dict

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_answer, answer):
    gt_answer = answer
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def create_demo_text(n_shot=6, cot_flag=True, shuffle=False):
    question, chain, answer = [], [], []
    question.append("Do hamsters provide food for any animals?")
    chain.append("Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals.")
    answer.append("yes")

    question.append("Could Brooke Shields succeed at University of Pennsylvania?")
    chain.append("Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania.")
    answer.append("yes")

    question.append("Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?")
    chain.append("Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5.")
    answer.append("no")

    question.append("Yes or no: Is it common to see frost during some college commencements?")
    chain.append("College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements.")
    answer.append("yes")

    question.append("Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?")
    chain.append("The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam.")
    answer.append("no")

    question.append("Yes or no: Would a pear sink in water?")
    chain.append("The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float.")
    answer.append("no")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text

def build_prompt(input_text, n_shot, cot_flag, shuffle):
    demo = create_demo_text(n_shot, cot_flag, shuffle)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def clean_answer(model_pred, random_guess=False):
    model_pred = model_pred.lower()

    if "Thus, yes." in model_pred:
        preds = "yes"
    elif SHORT_ANSWER_TRIGGER.lower() in model_pred:
        preds = model_pred.split(SHORT_ANSWER_TRIGGER.lower())[1].split(".")[0].strip()
    else:
        print("Warning: answer trigger not found in model prediction:", model_pred, "; returning yes/no based on exact match of `no`.", flush=True)
        if random_guess:
            preds = "no" if "no" in model_pred else "yes"
        else:
            return None
    if preds not in ["yes", "no"]:
        print("Warning: model prediction is not yes/no:", preds, "; returning no", flush=True)
        if random_guess:
            preds = "no"
        else:
            return None

    return (preds == "yes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Specifies the model you want to use")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-path", type=str, default="./results/strqa_result.json")
    parser.add_argument("--early-exit-layers", type=str, choices=['-1','low','high'], default="low", help="'-1' for greedy decoding. low/high for DoLa/ActLCD")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--relative_top", type=float, default=0.1)
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--loaded-model-name", type=str, default="./model/llama_bcq_strqa-low.pth", help="Only for ActLCD: Path to the BCQ model, otherwise set as '-1'. Naming logic: <MODEL NAME>_bcq_<BENCHMARK>-<PREMATURE LAYER>.pth")
    
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    llm = args.model_name.split('/')[1].split('-')[0]

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to

    # Get test file
    '''
    The StrategyQA dataset includes the followings files:
        strategyqa_train.json: The training set of StrategyQA, which includes 2,290 examples.
        strategyqa_train_paragraphs.json: Paragraphs from our corpus that were matched as evidence for examples in the training set.
        strategyqa_train_filtered.json: 2,821 additional questions, excluded from the official training set, that were filtered by our solvers during data collection (see more details in the paper).
        strategyqa_test.json: The test set of StrategyQA, which includes 490 examples.
    Here we only need the test set.
    '''
    fp = "./eva_dataset/strqa/strategyqa_train.json"
    if not os.path.exists(fp):
        download_url(
            'https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip', args.data_path)

        # Once the file is downloaded, unzip it
        with zipfile.ZipFile(os.path.join(args.data_path, 'strategyqa_dataset.zip'), 'r') as zip_ref:
            zip_ref.extractall(args.data_path)

    list_data_dict = load_jsonl(fp)
    
    if args.debug:
        list_data_dict = list_data_dict[:5]
    
    llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory, classifier_path=args.loaded_model_name)
    stop_word_list = ["Q:", "\n\n##"]
    llm.set_stop_words(stop_word_list)
    dola_layers = ''
    if args.early_exit_layers =='low':
        mode = "dola"
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
        dola_layers = 'low'
    elif args.early_exit_layers =='high':
        mode = "dola"
        if args.repetition_penalty is None:
            args.repetition_penalty = 1.2
        dola_layers = 'high'
    else:
        early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]
        if len(early_exit_layers) == 1:
            mode = "baseline"
            mature_layer = None
            premature_layer = None
            candidate_premature_layers = None
            if args.repetition_penalty is None:
                args.repetition_penalty = 1.0
        elif len(early_exit_layers) == 2:
            mode = "dola-static"
            mature_layer = early_exit_layers[1]
            premature_layer = early_exit_layers[0]
            candidate_premature_layers = None
            if args.repetition_penalty is None:
                args.repetition_penalty = 1.2
        else:
            mode = "dola"
            mature_layer = early_exit_layers[-1]
            premature_layer = None
            candidate_premature_layers = early_exit_layers[:-1]
            premature_layer_dist = {l:0 for l in candidate_premature_layers}
            if args.repetition_penalty is None:
                args.repetition_penalty = 1.2
    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': []}
    if llm.classifier is not None:
        mode = 'actlcd'
    print(f'MODE: Decoding as: {mode}')
    for sample in tqdm(list_data_dict):
        model_answer = None
        input_text = build_prompt(sample['question'], N_SHOT, COT_FLAG, args.do_shuffle)
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, relative_top=args.relative_top, loaded_model_name=args.loaded_model_name, dola_layers=dola_layers)
        if mode=='baseline':
            generate_kwargs = dict(max_new_tokens=args.max_new_tokens, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature, repetition_penalty=args.repetition_penalty, mode=mode, relative_top=args.relative_top)
        model_completion, c_dist = llm.generate(input_text, **generate_kwargs)
        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        model_completion = model_completion.strip()
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample['answer'])
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_answer'].append(model_answer)
        result_dict['model_completion'].append(model_completion)
        result_dict['full_input_text'].append(input_text)
        if args.debug:
            print(f'Full input_text:\n{input_text}\n\n')
            print(f'Question: {sample["question"]}\n\n'
                f'Answers: {sample["answer"]}\n\n'
                f'Model Answers: {model_answer}\n\n'
                f'Model Completion: {model_completion}\n\n'
                f'Is correct: {is_cor}\n\n')
        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.')

    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
    print(f"{float(sum(answers))/len(answers)}")