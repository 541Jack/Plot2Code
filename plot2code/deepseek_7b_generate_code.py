import sys
import os
import json
from tqdm import tqdm
from .utils import (
    get_parser,
    get_save_path,
    direct_prompt,
    read_jsonl_file,
    extract_code
)
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# Initialize the parser and arguments
parser = get_parser()
args = parser.parse_args()
image_directory = args.image_directory

# Specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Get current execution path
current_path = sys.path[0]
save_path = get_save_path(args)

previous_results = None
previous_filename = []

# Check if the save_path is empty. If it is not empty, load the already generated results
if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
    previous_results = read_jsonl_file(save_path)
    previous_filename = [result['question_id'] for result in previous_results]

if args.instruct:
    instructions = read_jsonl_file('data/ground_truth_code_with_instruction.jsonl')

# Iterate through all the PNG files in the image directory
if __name__ == '__main__':
    with open(save_path, "w") as jsonl_file:
        # Write the previous results to the file
        if previous_results is not None:
            for result in previous_results:
                jsonl_file.write(json.dumps(result) + '\n')
                jsonl_file.flush()

        for filename in tqdm(os.listdir(image_directory)):
            if filename in previous_filename:
                continue

            if filename.endswith(".png"):
                image_path = os.path.join(current_path, image_directory, filename)

                if args.instruct:
                    prompt = instructions[int(filename.rstrip('.png').lstrip('ground_truth_image_'))]['instruction'] + '\n' + direct_prompt
                else:
                    prompt = direct_prompt

                # Prepare the conversation
                conversation = [
                    {
                        "role": "User",
                        "content": "<image_placeholder>" + prompt,
                        "images": [image_path]
                    },
                    {
                        "role": "Assistant",
                        "content": ""
                    }
                ]

                # Load images and prepare inputs
                pil_images = load_pil_images(conversation)
                prepare_inputs = vl_chat_processor(
                    conversations=conversation,
                    images=pil_images,
                    force_batchify=True
                ).to(vl_gpt.device)

                # Run image encoder to get the image embeddings
                inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                # Run the model to get the response
                outputs = vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True
                )

                answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                generated_code = extract_code(answer.strip())

                jsonl_file.write(json.dumps({
                    'code': generated_code,
                    'question_id': filename,
                    'ground_truth_path': image_path
                }) + "\n")
                jsonl_file.flush()
