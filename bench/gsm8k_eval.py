'''
using 
lm_eval --model hf --model_args pretrained=/home/scc/OOM/mathfusion-ft-results --tasks gsm8k --device cuda:3 --batch_size auto -f 8 --seed 42
lm_eval --model hf --model_args pretrained=/home/scc/OOM/mathfusion-ft-results --tasks minerva_math --device cuda:2 --batch_size auto --seed 42

This is a tmp implement to evaluate the model on the GSM8K dataset using a 8-shot prompt.
'''

import torch
import re
import os
import argparse
import random
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    StoppingCriteria,
)
# Define a stopping condition for generation
class SpecificStringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, input_len):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[self.input_len:]
        
        return any(stop_string in current_text for stop_string in self.stop_strings)


def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        return None

def extract_ground_truth(text):
    return text.split('####')[-1].strip()

from datasets import load_dataset
from collections import Counter
import json


FEW_SHOT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: {question}
A:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/home/scc/OOM/mathfusion-ft-results')
    parser.add_argument('--use_majority_vote', action='store_true')
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--n_votes', type=int, default=1)
    args = parser.parse_args()


    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    print('Loading model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda:0', attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16) 
    
    print('\nLoading dataset...')
    dataset = load_dataset('gsm8k', "main", split='test')
    datasize = len(dataset)
    print('gsm8k test size:', datasize) 

    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>"
    ]

    results = []
    for i in tqdm(range(datasize), desc='Evaluating'):
        example = dataset[i]
        input_text = FEW_SHOT_PROMPT.format(question=example['question'])
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        ground_truth_answer = extract_ground_truth(example['answer'])

        stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(input_text))
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        model_answers = []
        if args.use_majority_vote:
            for _ in range(args.n_votes):
                with torch.no_grad():
                    outputs = model.generate(**inputs, temperature=args.temp, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract the final answer from the model's output
                output_text = output_text.split("A:")[-1].strip() 
                model_answer = extract_predicted_answer(output_text)
                model_answers.append({'text': output_text, 'numeric': model_answer})
        else:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria_list)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.split("A:")[-1].strip() 
            model_answer = extract_predicted_answer(output_text)
            model_answers.append({'text': output_text, 'numeric': model_answer})

        numeric_answers = [ma['numeric'] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]
        majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None

        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        results.append({
            'question': example['question'],
            'gold_answer_text': example['answer'],
            'model_answers_text': [ma['text'] for ma in model_answers],
            'extracted_model_answers': numeric_answers,
            'extracted_gold_answer': ground_truth_answer,
            'majority_answer': majority_answer,
            'correct': correct
        })
    
    cnt = 0
    for result in results:
        if result['correct']:
            cnt += 1
    total = len(results)
    print(f"Accuracy: {cnt} / {total} = {cnt / total :.4f}")

    results.append({'accuracy': cnt / total})

    os.makedirs('eval_results/few_shot', exist_ok=True)
    
    model_name = args.model.split('/')[-1]
    result_file = f"eval_results/few_shot/{model_name}"
    if args.use_majority_vote:
        result_file += f"_maj1@{args.n_votes}_temp{args.temp}"
    result_file += "_results.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")
                

if __name__ == '__main__':
    main()

