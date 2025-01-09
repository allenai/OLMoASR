"""
Code to add embedding scores.

Will operate on a jsonl (on weka), and put a jsonl output to s3


"""

import argparse
import json
import gzip
import boto3 
import torch
import os
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import webvtt


HUMAN_TEXT_KEY = 'content'
MACH_TEXT_KEY = 'content_mach'


# =================================================================
# =                      I/O STUFf                                =
# =================================================================

def load_jsonl_gz(input_path):
	lines = gzip.decompress(open(input_path, 'rb').read()).splitlines()
	return [json.loads(_) for _ in lines]



def save_jsonl_gz(dict_list, output_path):
	jsonl_content = gzip.compress(b'\n'.join(json.dumps(_).encode('utf-8') for _ in dict_list))
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, 'wb') as f:
		f.write(jsonl_content)
	


# =================================================================
# =                     CONTENT PROCESSING STUFF                  =
# =================================================================

def collect_machine_text(machine_text):
    """ The way the youtube captioner generally works is like
    if lines are like ABCD
    then captions are like 
    A
    A, B
    B
    B, C
    C
    C, D
    
    So to recover ABCD, we can:
    1. only look at captions with a newline
    2. take the full first one
    3. take the second line of every other one

    ^ BUT ONLY FOR GROUPS WHERE THERE ARE CONTIGUOUS TIMESTAMPS
    """
    vtt_captions = list(webvtt.from_string(machine_text))    
    caption_groups = []
    cur = []
    for c in vtt_captions:
        if cur == [] or c.start == cur[-1].end:
            cur.append(c)
        else:
            caption_groups.append(cur)
            cur = []
    if cur:
        caption_groups.append(cur)

    
    def process_subgroup(subgroup):
        vtt_text = [_.text for _ in subgroup]    
        newline_counts = [_.count('\n') for _ in vtt_text]
        assert binary_alternates(newline_counts), AssertionError("Weird machine transcript format")       
        newline_captions = [_ for _ in vtt_text if _.count('\n') > 0]
        full_text = [newline_captions[0]]
        for i in range(1, len(newline_captions)):
            full_text.append(newline_captions[i].split('\n')[-1])        
        return full_text

    all_groups = []
    for group in caption_groups:
        all_groups.extend(process_subgroup(group))
    return '\n'.join(all_groups).replace('\n', ' ')


def binary_alternates(els):
    if len(els) == 0:
        return True
    so_far = True
    cur = els[0]
    for i in range(len(els) - 1):
        so_far = so_far and (1 - els[i+1] == els[i])
    return so_far
    

def collect_human_text(human_text):
    """ Here there should be no repeated text captions... 
    """
    vtt_text = [_.text for _ in webvtt.from_string(human_text)]
    return '\n'.join(vtt_text).replace('\n', ' ')    




# =================================================================
# =                     EMBEDDING/SCORING STUFF                   =
# =================================================================

@torch.no_grad
def embed_text(tokenizer, model, texts):
	inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
	if torch.cuda.is_available():
		inputs = {k: v.cuda() for k,v in inputs.item()}

	outputs = model(**inputs)
	embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
	return embeddings


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

@torch.no_grad()
def batch_cosine_distance(a, b):
	return torch.nn.functional.cosine_similarity(a, b).cpu().numpy()

# =================================================================
# =                      MAIN BLOCK                               =
# =================================================================


def main(input_jsonl, output_jsonl, batch_size=256):

	# Load data
	input_dicts = load_jsonl_gz(input_jsonl)
	proc_texts = {}
	d_by_id = {}
	for d in input_dicts:
		d_id = d['id']
		try:
			human_text = collect_human_text(d[HUMAN_TEXT_KEY])
			mach_text = collect_machine_text(d[MACH_TEXT_KEY])
			proc_texts[d_id] = (human_text, mach_text)
			d_by_id[d_id] = d
		except Exception as err:
			print(d_id, err)
			pass
	proc_texts = list(proc_texts.items())
	print("Got machine texts from %s/%s of lines" % (len(proc_texts), len(input_dicts)))

	# Load model
	tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
	model = AutoModel.from_pretrained('facebook/contriever-msmarco')	
	if torch.cuda.is_available():
		model = model.cuda()



	# Make batches:
	id_batch_size = max(batch_size // 2, 1)
	num_batches = (len(proc_texts) - 1) // id_batch_size + 1

	# Iterate over batches: embed, score, and then assign scores
	for batch_num in tqdm(range(num_batches)):
		batch = proc_texts[batch_num * id_batch_size : (batch_num + 1) * id_batch_size]
		batch_texts = [v[0] for k,v in batch] + [v[1] for k,v in batch]

		embeddings = embed_text(tokenizer, model, batch_texts)
		scores = batch_cosine_distance(*torch.split(embeddings, len(batch)))

		for item, score in zip(batch, scores):
			d_id = item[0]
			d_by_id[d_id]['man_mach_score'] = float(score)
		output_dicts = list(d_by_id.values())

	# Save output
	save_jsonl_gz(output_dicts, output_jsonl)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--input', required=True, help='s3 URI to jsonl.gz we process')
	parser.add_argument('--output', required=True, help='s3 URI where the output jsonl.gz file should go')
	parser.add_argument('--batch-size', type=int, default=256)

	args = parser.parse_args()
	main(args.input, args.output, args.batch_size)




