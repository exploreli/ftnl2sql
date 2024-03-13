
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
def get_table_name(str):
    return str.split(" ")[2]

import json

model_path="e:\\code\\chatglm\\chatglm2"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
 
model = PeftModel.from_pretrained(model, "model-lora")#
model=model.eval() 
with open("test_data",encoding="utf-8") as f:
    lines=[ json.loads(s.strip())for s in f.readlines()]
for example in lines:
    table_name=get_table_name(example['context'])
    input='context : '+example["context"].replace(table_name,"T")+"qusetion :"+example['question']
    print (input)
    response, history = model.chat(tokenizer, input, history=[])
    print(response)
    print ("\n\n\n")

 

