
from transformers import BertTokenizerFast, GPT2LMHeadModel
import torch

tokenizer_gpt3 = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")        
model_gpt3 = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")

# 입력 문장
input_text = "밥 뭐먹을래"
# 입력 문장을 토큰화하고 텐서로 변환
input_ids = tokenizer_gpt3.encode(input_text, return_tensors='pt')[0].unsqueeze(0)  # (1, seq_len)

# 모델에 입력하고 출력 생성
output = model_gpt3.generate(input_ids, max_length=50, num_return_sequences=1)

# 출력 토큰을 텍스트로 변환
output_text = tokenizer_gpt3.decode(output[0], skip_special_tokens=True)

print("모델 출력:", output_text)

'''
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="kykim/gpt3-kor-small_based_on_gpt2")
'''

'''
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
'''