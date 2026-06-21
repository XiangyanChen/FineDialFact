from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig, \
    AutoProcessor, Gemma3ForConditionalGeneration
import torch
import re
import gc
import os
from openai import OpenAI
import google.generativeai as genai
from google.generativeai import GenerativeModel


class LLM:
    """
    A unified interface for different language models including local and API-based models.
    Supports various model types: FLAN-T5, Llama, Gemma, Mistral, Qwen, ChatGLM, and API-based
    models like Llama3 (AWS Bedrock), DeepSeek, Gemini, and GPT models.
    """

    def __init__(self, model_id, train_ckpt_path=None, aws_region="us-east-1"):
        """
        Initialize the LLM wrapper.

        Args:
            model_id: Identifier for the model to load
            train_ckpt_path: Optional path to custom checkpoint
            aws_region: AWS region for Bedrock services
        """
        self.model = None
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device} (CUDA available: {torch.cuda.is_available()})')

        if model_id == "deepseek":
            self._init_deepseek()
        elif "Llama" in model_id:
            self._init_llama(model_id, train_ckpt_path)
        elif "gemma" in model_id.lower():
            self._init_gemma(model_id, train_ckpt_path)
        elif model_id == "gemini":
            self._init_gemini()
        elif "mistral" in model_id.lower():
            self._init_mistral(model_id, train_ckpt_path)
        elif "qwen" in model_id.lower() or "qwq" in model_id.lower():
            self._init_qwen(model_id, train_ckpt_path)
        elif "gpt" in model_id:
            self.client = OpenAI()
            pass

    def _init_deepseek(self):
        """Initialize DeepSeek API client"""
        self.api_key = os.environ.get("deepseek")
        self.base_url = "https://api.deepseek.com"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        print(f'Using DeepSeek API with base URL: {self.base_url}')

    def _init_llama(self, model_id, train_ckpt_path):
        """Initialize Llama models"""
        model_path = train_ckpt_path if train_ckpt_path else f"meta-llama/{model_id}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Apply custom chat template if using a custom checkpoint
        if train_ckpt_path:
            self.tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    def _init_gemma(self, model_id, train_ckpt_path):
        """Initialize Gemma models"""
        model_path = train_ckpt_path if train_ckpt_path else f"google/{model_id}"
        print(f"Loading Gemma model from {model_path}")

        # Use AutoProcessor for models that support images
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = getattr(self.processor, 'tokenizer', None) or AutoTokenizer.from_pretrained(model_path)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        ).eval()

        print("Gemma model loaded successfully")
        torch._dynamo.config.cache_size_limit = 256  # default is 64

    def _init_gemini(self):
        """Initialize Google Gemini API"""
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=self.api_key)

        # Default to Gemini-1.5-Pro, can be made configurable
        self.gemini_model_name = "gemini-1.5-pro"
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40
        }
        print(f'Using Google Gemini API with model: {self.gemini_model_name}')

    def _init_mistral(self, model_id, train_ckpt_path):
        """Initialize Mistral models"""
        model_path = train_ckpt_path if train_ckpt_path else f"mistralai/{model_id}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def _init_qwen(self, model_id, train_ckpt_path):
        """Initialize Qwen models"""
        model_path = train_ckpt_path if train_ckpt_path else f"Qwen/{model_id}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128, max_new_tokens=128,
                 system_prompt=None, temperature=1):
        """
        Generate text based on the provided prompt.

        Args:
            prompt: Input text or list of input texts
            sample_idx: Sample index for batch processing
            max_sequence_length: Maximum sequence length
            max_output_length: Maximum output length
            max_new_tokens: Maximum number of new tokens to generate
            system_prompt: Optional system prompt for instruction-tuned models

        Returns:
            List of generated text outputs
        """
        return self._generate(prompt, max_sequence_length=max_sequence_length,
                              max_output_length=max_output_length,
                              max_new_tokens=max_new_tokens,
                              system_prompt=system_prompt, temperature=temperature)

    def _generate(self, prompt_list, max_sequence_length=2048, max_output_length=128, max_new_tokens=128,
                  system_prompt=None, temperature=1):
        """
        Internal generate function that handles different model types.
        """
        # Convert single prompt to list if needed
        if not isinstance(prompt_list, list):
            prompt_list = [prompt_list]

        elif self.model_id == "deepseek":
            return self._generate_deepseek(prompt_list, max_new_tokens, system_prompt, temperature)
        elif self.model_id == "gemini":
            return self._generate_gemini(prompt_list, system_prompt, temperature)
        elif "gpt" in self.model_id:
            return self._generate_openai_gpt(prompt_list, max_new_tokens, system_prompt, temperature)
        else:
            # Local models
            return self._generate_local_model(prompt_list, max_sequence_length, max_output_length, max_new_tokens,
                                              system_prompt, temperature)

    def _generate_deepseek(self, prompt_list, max_new_tokens, system_prompt, temperature=1):
        """Generate text using DeepSeek API"""
        results = []
        for i, prompt in enumerate(prompt_list):
            messages = []

            # Add system prompt if provided
            if isinstance(system_prompt, list) and len(system_prompt) > 0:
                curr_system_prompt = system_prompt[i]
                if curr_system_prompt.strip() != "":
                    messages.append({"role": "system", "content": curr_system_prompt})
            elif isinstance(system_prompt, str) and system_prompt.strip() != "":
                messages.append({"role": "system", "content": system_prompt})

            # Add user message
            messages.append({"role": "user", "content": prompt})

            try:
                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    stream=False
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                results.append(f"Error with DeepSeek API: {str(e)}")

        return results

    def _generate_gemini(self, prompt_list, system_prompt, temperature=1):
        """Generate text using Google Gemini API"""
        results = []

        # Ensure system_prompt is a list with matching length
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt] * len(prompt_list)
        elif not isinstance(system_prompt, list) or len(system_prompt) != len(prompt_list):
            system_prompt = [""] * len(prompt_list)
        generation_config = {"temperature": temperature}
        for user_prompt, curr_system_prompt in zip(prompt_list, system_prompt):
            model = GenerativeModel(
                model_name=self.gemini_model_name,
                generation_config=generation_config
            )

            chat = model.start_chat(history=[])
            response = chat.send_message(
                f"{curr_system_prompt}\n\n{user_prompt}", stream=False
            )

            results.append(response.text)

        return results

    def _generate_openai_gpt(self, prompt_list, max_new_tokens, system_prompt, temperature=1):
        """Generate text using OpenAI GPT models"""
        # Prepare messages list
        messages_list = []
        for i, prompt in enumerate(prompt_list):
            messages = []
            if isinstance(system_prompt, list) and len(system_prompt) > 0:
                messages.append({"role": "system", "content": system_prompt[i]})
            elif system_prompt is not None and system_prompt != "":
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)

        # Select model based on model_id
        model = "gpt-4o" if self.model_id == "gpt" else "o4-mini"

        # Call API for each prompt
        responses = []
        for messages in messages_list:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            responses.append(response.choices[0].message.content)

        return responses

    def _generate_local_model(self, prompt_list, max_sequence_length, max_output_length, max_new_tokens, system_prompt, temperature=1):
        """Generate text using local models"""
        with torch.no_grad():
            if "Llama-3" in self.model_id:
                return self._generate_llama3(prompt_list, max_new_tokens, system_prompt, temperature)
            elif "Llama-2" in self.model_id:
                return self._generate_llama2(prompt_list, max_new_tokens, system_prompt, temperature)
            elif "gemma" in self.model_id.lower():
                return self._generate_gemma_model(prompt_list, max_new_tokens, system_prompt, temperature)
            elif "mistral" in self.model_id.lower():
                return self._generate_mistral_model(prompt_list, max_new_tokens, system_prompt, temperature)
            elif "qwen" in self.model_id.lower() or "qwq" in self.model_id.lower():
                return self._generate_qwen_model(prompt_list, max_new_tokens, system_prompt, temperature)


    def _generate_llama3(self, prompt_list, max_new_tokens, system_prompt, temperature=1):
        """Generate text using Llama-3 models"""
        # Format messages with system prompt if provided
        formatted_prompts = self._format_llama_prompts(prompt_list, system_prompt)

        # Apply chat template
        texts = self.tokenizer.apply_chat_template(formatted_prompts, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(texts, padding="longest", return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        # Generate
        outputs = self.model.generate(**inputs,
                                      max_new_tokens=max_new_tokens,
                                      min_length=2,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-7)  # Avoid division by zero
                                      )

        # Decode only the new tokens
        response = outputs[:, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.batch_decode(response, skip_special_tokens=True)

    def _generate_llama2(self, prompt_list, max_new_tokens, system_prompt, temperature=1):
        """Generate text using Llama-2 models"""
        template = "<s>[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]"

        # Format prompts with system prompt if provided
        if system_prompt is not None:
            if isinstance(system_prompt, list):
                formatted_prompts = [
                    template % (sp, prompt)
                    for sp, prompt in zip(system_prompt, prompt_list)
                ]
            else:
                formatted_prompts = [
                    template % (system_prompt, prompt)
                    for prompt in prompt_list
                ]
        else:
            formatted_prompts = [
                template % ("", prompt)
                for prompt in prompt_list
            ]

        # Tokenize and generate
        inputs = self.tokenizer(formatted_prompts, padding="longest", return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        outputs = self.model.generate(**inputs,
                                      max_new_tokens=max_new_tokens,
                                      min_length=2,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-7)  # Avoid division by zero
                                      )

        # Decode only the new tokens
        response = outputs[:, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.batch_decode(response, skip_special_tokens=True)

    def _generate_gemma_model(self, prompt_list, max_new_tokens, system_prompt, temperature=1):
        """Generate text using Gemma models"""
        # Ensure system_prompt is a list with matching length
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt] * len(prompt_list)

        # Format messages for Gemma
        processed_gemma_prompts = []
        for sp, prompt in zip(system_prompt, prompt_list):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sp}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            processed_gemma_prompts.append(messages)

        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            processed_gemma_prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        # Calculate input lengths to properly extract new tokens
        input_lens = (inputs["input_ids"] != self.processor.tokenizer.pad_token_id).sum(dim=1)

        # Generate
        with torch.inference_mode():
            generations = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-7)  # Avoid division by zero
            )

        # Decode only the new tokens for each prompt
        results = []
        for i, generation in enumerate(generations):
            generated_tokens = generation[input_lens[i]:]
            decoded = self.processor.decode(generated_tokens, skip_special_tokens=True)
            results.append(decoded)

        return results

    def _generate_mistral_model(self, prompt_list, max_new_tokens, system_prompt, temperature=1):
        """Generate text using Mistral models"""
        # Format messages with system prompt if provided
        formatted_prompts = self._format_llama_prompts(prompt_list, system_prompt)

        # Apply chat template and generate
        texts = self.tokenizer.apply_chat_template(formatted_prompts, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(texts, padding="longest", return_tensors="pt")
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        outputs = self.model.generate(**inputs,
                                      max_new_tokens=max_new_tokens,
                                      min_length=2,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-7))  # Avoid division by zero

        # Decode only the new tokens
        response = outputs[:, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.batch_decode(response, skip_special_tokens=True)

    def _generate_qwen_model(self, prompt_list, max_new_tokens, system_prompt, temperature=1):
        """Generate text using Qwen models"""
        # Format messages with system prompt if provided
        messages_list = []
        for i, prompt in enumerate(prompt_list):
            messages = []
            if system_prompt is not None:
                if isinstance(system_prompt, list) and len(system_prompt) > 0:
                    messages.append({"role": "system", "content": system_prompt[i]})
                else:
                    messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)

        # Generate for each prompt
        results = []
        for messages in messages_list:
            texts = self.tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = self.tokenizer(texts, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs,
                                          max_new_tokens=max_new_tokens,
                                          min_length=2,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          do_sample=temperature > 0,
                                          temperature=max(temperature, 1e-7))  # Avoid division by zero)

            # Decode only the new tokens
            response = outputs[0][inputs["input_ids"].shape[1]:]
            result = self.tokenizer.decode(response, skip_special_tokens=True)
            results.append(result)
        return results

    def _format_llama_prompts(self, prompt_list, system_prompt):
        """Helper function to format prompts for Llama/Mistral models"""
        if system_prompt is not None and ((isinstance(system_prompt, list) and len(system_prompt) > 0) or
                                          (isinstance(system_prompt, str) and system_prompt.strip() != "")):
            if isinstance(system_prompt, list):
                return [[{"role": "system", "content": sp}, {"role": "user", "content": prompt}]
                        for sp, prompt in zip(system_prompt, prompt_list)]
            else:
                return [[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                        for prompt in prompt_list]
        else:
            return [[{"role": "user", "content": prompt}] for prompt in prompt_list]

    def remove_not_ascii(self, text):
        """Remove non-ASCII characters from text"""
        pattern = re.compile(r'[^\x00-\x7F]')
        return re.sub(pattern, ' ', text)

    def destroy(self):
        """Clean up resources and free memory"""
        if self.model:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
