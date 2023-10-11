import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

pretrained_model = 'ramsrigouthamg/t5-large-paraphraser-diverse-high-quality'

model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, legacy=False)

print(f'Loaded model: {pretrained_model}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device.type)
model = model.to(device)

# Diverse Beam search
context = 'Well-experienced in the programming language Python and data \
    preprocessing, and cleared theoretical concepts of AI and machine learning'

text = 'paraphrase: ' + context + ' </s>'

encoding = tokenizer.encode_plus(
    text, max_length=128, padding='max_length', return_tensors='pt')
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

model.eval()
diverse_beam_outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_mask,
    max_length=128,  # 128
    early_stopping=True,  # True
    num_beams=5,  # 5
    num_beam_groups=5,  # 5
    num_return_sequences=5,  # 5
    diversity_penalty=0.70  # 0.70
)

print('Original: ', context)
for beam_output in diverse_beam_outputs:
    sent = tokenizer.decode(
        beam_output,
        skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(sent)
