import warnings
from abc import ABC
from typing import Optional, List, Any, Dict, Mapping

import torch
import transformers
from PIL import Image
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "BAAI/Bunny-Llama-3-8B-V"

transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')


class BunnyLlama3(LLM, ABC):
    model_name: str = MODEL_NAME
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    device: str = 'cuda:0'

    model: Any = None
    tokenizer: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            'BAAI/Bunny-Llama-3-8B-V',
            torch_dtype=torch.float16,  # float32 for cpu
            device_map='auto',
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            'BAAI/Bunny-Llama-3-8B-V',
            trust_remote_code=True
        )

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        # text prompt
        text = f"Give a helpful, detailed answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(
            text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long
        ).unsqueeze(0).to(self.device)

        # image, sample images can be found in images folder
        image = Image.open(kwargs.get('img_url'))
        image_tensor = self.model.process_images(
            [image], self.model.config
        ).to(dtype=self.model.dtype, device=self.device)

        # generate
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=100,
            use_cache=True)[0]

        output = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

        return output

    def _llm_type(self) -> str:
        return 'bunny_llama3'


if __name__ == '__main__':
    llm = BunnyLlama3()

    question = 'Why is the image funny? Answer with chinese.'
    result = llm.invoke(question, img_url='image/test.jpg')

    print(result)
