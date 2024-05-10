import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

# disable some warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')

# set device
device = 'cuda:0'  # or cpu
torch.set_default_device(device)

# create model
model = AutoModelForCausalLM.from_pretrained(
    'BAAI/Bunny-Llama-3-8B-V',
    torch_dtype=torch.float16,  # float32 for cpu
    device_map='auto',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    'BAAI/Bunny-Llama-3-8B-V',
    trust_remote_code=True
)

# text prompt
prompt = 'Why is the image funny? Answer with chinese.'
text = f"Give a helpful, detailed answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)

# image, sample images can be found in images folder
image = Image.open('BunnyDemo/image/test.jpg')
image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)

# generate
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    max_new_tokens=100,
    use_cache=True)[0]

print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
