"""
Evaluate model's streaming capability and text generation quality with Distributed Data Parallel.
This script tests:
1. Frame decision capability: Model's ability to decide when to stop inputting frames after seeing the query
2. Text generation quality: Quality of the generated answer

The model generates "," or "\n" after each frame to decide:
- "," : continue to next frame
- "\n" : stop frame input and generate response
"""


import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import base64
from io import BytesIO
from qwen_vl_utils import extract_vision_info

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from transformers import AutoProcessor, AutoTokenizer
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
from qwen_vl.model.geometry_encoders import (
    patch_geometry_encoder_with_noise,
    swap_geometry_encoder,
)
from qwen_vl.data.utils import load_and_preprocess_images


class StreamingEvaluator:
    """Evaluator for model's streaming capability with DDP support"""
    
    def __init__(
        self,
        model_path: str,
        local_rank: int = -1,
        world_size: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ):
        """
        Initialize evaluator with DDP support
        
        Args:
            model_path: Path to the model checkpoint
            local_rank: Local rank for distributed training (-1 for single GPU)
            world_size: Total number of processes
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 for greedy)
            use_gradient_checkpointing: Whether to use gradient checkpointing to save memory
        """
        self.local_rank = local_rank
        self.world_size = world_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if local_rank == -1:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
        
        print(f"Loading model from {model_path}...")
        print(f"Local rank: {local_rank}, World size: {world_size}")
        print(f"Gradient checkpointing: {use_gradient_checkpointing}")

        self.model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            attn_implementation="flash_attention_2",
        )

        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled (slower but memory-efficient)")

        self.model = self.model.to(self.device)
        self.model.eval()

        if local_rank != -1:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            padding_side="left"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left"
        )
        
        # Get special token IDs
        model_config = self.model.module.config if hasattr(self.model, 'module') else self.model.config
        self.image_token_id = model_config.image_token_id
        self.vision_start_token_id = getattr(model_config, 'vision_start_token_id', None)
        self.vision_end_token_id = getattr(model_config, 'vision_end_token_id', None)
        self.frame_interval_token_id = self.tokenizer.encode(',', add_special_tokens=False)[0]
        self.frame_end_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]

        # Chat-template related token IDs
        self.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.im_start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.assistant_token_ids = self.tokenizer.encode("assistant", add_special_tokens=False)
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
    
        if self.is_main_process():
            print(f"Model loaded on GPU {local_rank}")
    
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.local_rank <= 0
    
    def parse_conversation_with_query(self, conversation_value: str) -> Tuple[List[str], str, List[str]]:
        """
        Parse conversation to extract images before query, query text, and images after query
        """
        parts = conversation_value.split('<image>')
        
        query_idx = -1
        query_text = ""
        
        for i, part in enumerate(parts):
            cleaned = part.strip().strip(',').strip()
            if cleaned and not all(c in '.,\n ' for c in cleaned):
                query_idx = i
                query_text = cleaned.rstrip('.')
                break
        
        if query_idx == -1:
            raise ValueError(f"Could not find query text in conversation: {conversation_value}")
        
        images_before = ['<image>'] * (query_idx)
        images_after = ['<image>'] * (len(parts) - query_idx - 1)
        
        return images_before, query_text, images_after
    
    def _print_memory_stats(self, stage: str):
        """Print current GPU memory usage"""
        if self.is_main_process() and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"  [{stage}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        base64_image = image.convert("RGB")
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")
        return f"data:image/jpeg;base64,{base64_string}"
    
    def _build_soley_image_content(self, images: List[Image.Image]) -> List[Dict]:
        """Build message content with images"""
        content = []
        for image in images:
            base64_string = self._encode_image_to_base64(image)
            content.append({
                "type": "image",
                "image": base64_string
            })
        return content

    def _build_image_add_comma_content(self, images: List[Image.Image]) -> List[Dict]:
        """Build message content with images and commas"""
        content = []
        for image in images:
            base64_string = self._encode_image_to_base64(image)
            content.append({
                "type": "image",
                "image": base64_string
            })
            content.append({
                "type": "text",
                "text": ","
            })
        return content
    
    def generate_with_streaming_decision(
        self,
        all_images: List[Image.Image],
        query_text: str,
        num_images_before_query: int,
    ) -> Tuple[int, str, Dict[str, float]]:
        """
        Generate with streaming frame decision using KV cache
        
        Returns:
            Tuple[int, str, Dict[str, float]]: 
                - last_frame_idx: index of the last frame used
                - generated_text: the generated response
                - timing_stats: dictionary containing timing information
        """

        images_before = all_images[:num_images_before_query]
        images_after = all_images[num_images_before_query:]
        
        if len(images_after) == 0:
            raise ValueError("No images after query for streaming decision!")
        
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        
        first_image_after = images_after[0]
        remaining_images_after = images_after[1:]

        # ===== Step 1: Build initial messages =====
        content = []
        if images_before:
            content = self._build_image_add_comma_content(images_before)
            content.append({"type": "text", "text": f"{query_text}"})
            first_image_content = self._build_soley_image_content([first_image_after])
            content.extend(first_image_content)
        else:
            content = [{"type": "text", "text": f"{query_text},"}]
            first_image_content = self._build_soley_image_content([first_image_after])
            content.extend(first_image_content)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        text = text[:-len("<|im_end|>")-1]

        # ===== Step 2: Process initial images =====
        geometry_encoder_inputs = []
        image_inputs = []
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        
        vision_info = extract_vision_info([messages[-1]])
        for ele in vision_info:
            if "image" in ele:
                image_str = ele["image"]
                if isinstance(image_str, str) and "base64," in image_str:
                    _, base64_data = image_str.split("base64,", 1)
                    data = base64.b64decode(base64_data)
                    with BytesIO(data) as bio:
                        image = Image.open(bio).copy()
                else:
                    raise ValueError(f"Unexpected image format")
                
                image_tensor = load_and_preprocess_images([image])[0]
                geometry_encoder_inputs.append(image_tensor.clone())
                
                _, height, width = image_tensor.shape
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                
                image_tensor = image_tensor[:, :height, :width]
                image_inputs.append(image_tensor)
       
        self._print_memory_stats("Before processor")
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
                do_rescale=False
            ).to(self.device)
            
        if hasattr(model_to_use.config, 'use_geometry_encoder') and model_to_use.config.use_geometry_encoder:
            inputs['geometry_encoder_inputs'] = [torch.stack(geometry_encoder_inputs).to(self.device)]
        
        del image_inputs
        torch.cuda.empty_cache()
        
        self._print_memory_stats("After processor")

        # ===== Step 3: First forward pass =====
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        pixel_values = inputs.get('pixel_values', None)
        image_grid_thw = inputs.get('image_grid_thw', None)
        geometry_encoder_inputs_list = inputs.get('geometry_encoder_inputs', None)

        # Track the current sequence length (used to compute cache_position).
        current_seq_len = input_ids.shape[1]
        
        past_key_values = None
        
        self._print_memory_stats("Before first forward")
        
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if self.is_main_process():
                print("Starting first forward pass...")

            outputs = model_to_use(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                geometry_encoder_inputs=geometry_encoder_inputs_list,
                past_key_values=None,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            del outputs, inputs
            torch.cuda.empty_cache()
            
            self._print_memory_stats("After first forward")
            
            # Sample next token
            if self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            next_token_id = next_token[0, 0].item()
            
            if self.is_main_process():
                print(f"  Generated token ID in the first generation: {next_token_id}")
                print(f"  Decoded token: '{str(self.tokenizer.decode([next_token_id]))}'")

        # ===== Helper: generate the final response =====
        def generate_final_response(past_key_values, current_seq_len) -> Tuple[str, Dict[str, float]]:
            """Generate the final text response and return timing info."""
            if self.is_main_process():
                print(f"\n{'='*70}")
                print(f"Generating final response")
                print(f"{'='*70}")
            
            self._print_memory_stats("Before final generation")

            answer_generation_start_time = time.perf_counter()
            ttft = None  # Time to first token

            chat_prompt_tokens = [
                self.im_end_token_id,
                self.newline_token_id,
                self.im_start_token_id,
            ] + self.assistant_token_ids + [self.newline_token_id]
            
            chat_prompt_tensor = torch.tensor(
                [chat_prompt_tokens],
                dtype=torch.long,
                device=self.device
            )
            
            # Forward the chat prompt to update the KV cache.
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                cache_position = torch.arange(
                    current_seq_len, 
                    current_seq_len + chat_prompt_tensor.shape[1],
                    device=self.device
                )

                outputs = model_to_use(
                    input_ids=chat_prompt_tensor,
                    attention_mask=None,  # not needed when using KV cache
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                current_seq_len += chat_prompt_tensor.shape[1]
                del outputs

            # Manual decoding loop.
            generated_tokens: List[int] = []

            for step in range(self.max_new_tokens):
                if step == 0:
                    # On the first step, re-forward the last prompt token to obtain its logits.
                    last_token = chat_prompt_tensor[:, -1:]
                    cache_position = torch.tensor([current_seq_len - 1], device=self.device)
                else:
                    last_token = next_token
                    cache_position = torch.tensor([current_seq_len], device=self.device)
                
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model_to_use(
                        input_ids=last_token,
                        attention_mask=None,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        use_cache=True,
                    )
                    logits = out.logits[:, -1, :]
                    past_key_values = out.past_key_values

                if self.temperature > 0:
                    probs = torch.softmax(logits / self.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                next_token_id = next_token.item()

                # Record TTFT (time to first token).
                if step == 0:
                    ttft = time.perf_counter() - answer_generation_start_time

                generated_tokens.append(next_token_id)
                current_seq_len += 1

                if next_token_id == self.tokenizer.eos_token_id:
                    break
                if next_token_id == self.im_end_token_id:
                    break

            answer_generation_end_time = time.perf_counter()
            answer_generation_latency = answer_generation_end_time - answer_generation_start_time
            
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            num_tokens = len(generated_tokens)
            tokens_per_second = num_tokens / answer_generation_latency if answer_generation_latency > 0 else 0

            timing_stats = {
                'ttft': ttft if ttft is not None else 0.0,
                'answer_generation_latency': answer_generation_latency,
                'num_output_tokens': num_tokens,
                'tokens_per_second': tokens_per_second,
            }
            
            if self.is_main_process():
                print(f"\nFinal Generated Response:\n{generated_text}\n")
                print(f"  TTFT: {ttft*1000:.2f}ms" if ttft else "  TTFT: N/A")
                print(f"  Answer Generation Latency: {answer_generation_latency*1000:.2f}ms")
                print(f"  Output Tokens: {num_tokens}")
                print(f"  Tokens/sec: {tokens_per_second:.2f}")
            
            return generated_text, timing_stats

        # ===== Step 4: Handle the Frame 0 decision =====
        current_seq_len += 1  # account for the just-generated decision token

        # Forward the decision token to update the KV cache.
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            cache_position = torch.tensor([current_seq_len - 1], device=self.device)
            outputs = model_to_use(
                input_ids=next_token,
                attention_mask=None,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            del outputs
        
        if next_token_id == self.frame_end_token_id:
            if self.is_main_process():
                print(f"  Decision: STOP (\\n)")
            generated_text, timing_stats = generate_final_response(past_key_values, current_seq_len)
            return 0, generated_text, timing_stats
        
        elif next_token_id == self.frame_interval_token_id:
            if self.is_main_process():
                print(f"  Decision: CONTINUE (,)")
            
        else:
            if self.is_main_process():
                print(f"  Decision: UNEXPECTED ('{self.tokenizer.decode([next_token_id])}'), treating as STOP")
            generated_text, timing_stats = generate_final_response(past_key_values, current_seq_len)
            return 0, generated_text, timing_stats

        # ===== Step 5: Process the remaining frames one by one =====
        current_frame_index = 0
        timing_stats = None
        
        for frame_idx, image in enumerate(remaining_images_after, start=1):
            if self.is_main_process():
                print(f"\n{'='*70}")
                print(f"Processing frame {frame_idx} after query")
                print(f"{'='*70}")
            
            self._print_memory_stats(f"Before frame {frame_idx}")

            new_image_inputs = []
            new_geometry_encoder_inputs = []
            
            image_tensor = load_and_preprocess_images([image])[0]
            new_geometry_encoder_inputs.append(image_tensor.clone())
            
            _, height, width = image_tensor.shape
            if (width // patch_size) % merge_size > 0:
                width = width - (width // patch_size) % merge_size * patch_size
            if (height // patch_size) % merge_size > 0:
                height = height - (height // patch_size) % merge_size * patch_size
            
            image_tensor = image_tensor[:, :height, :width]
            new_image_inputs.append(image_tensor)

            image_text = "<|vision_start|><|image_pad|><|vision_end|>"
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                new_image_processor_inputs = self.processor(
                    text=[image_text],
                    images=new_image_inputs,
                    videos=None,
                    return_tensors="pt",
                    do_rescale=False
                ).to(self.device)
            
            del new_image_inputs
            torch.cuda.empty_cache()
            
            new_image_token_tensor = new_image_processor_inputs['input_ids']
            new_pixel_values = new_image_processor_inputs['pixel_values']
            new_image_grid_thw = new_image_processor_inputs['image_grid_thw']
            new_attention_mask = new_image_processor_inputs['attention_mask']

            new_geo_tensor = torch.stack(new_geometry_encoder_inputs).to(self.device)

            num_new_tokens = new_image_token_tensor.shape[1]

            # Forward the new frame (visual inputs must be passed because this is a fresh image).
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                cache_position = torch.arange(
                    current_seq_len, 
                    current_seq_len + num_new_tokens,
                    device=self.device
                )
                
                past_len = past_key_values[0][0].shape[-2]
                # History mask filled with ones (every cached token is visible).
                past_mask = torch.ones(
                    (new_attention_mask.shape[0], past_len), 
                    dtype=new_attention_mask.dtype, 
                    device=self.device
                )

                # Concatenate history mask with the new-image mask into the full mask.
                full_attention_mask = torch.cat([past_mask, new_attention_mask], dim=1)

                outputs = model_to_use(
                    input_ids=new_image_token_tensor,
                    attention_mask=full_attention_mask,
                    pixel_values=new_pixel_values,
                    image_grid_thw=new_image_grid_thw,
                    geometry_encoder_inputs=[new_geo_tensor],
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                current_seq_len += num_new_tokens
                
                del outputs, new_image_processor_inputs
                torch.cuda.empty_cache()
                
                self._print_memory_stats(f"After frame {frame_idx} forward")
                
                if self.temperature > 0:
                    probs = torch.softmax(logits / self.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                next_token_id = next_token[0, 0].item()
                
                if self.is_main_process(): 
                    print(f"  Generated token ID: {next_token_id}")
                    print(f"  Decoded token: '{self.tokenizer.decode([next_token_id])}'")

            current_frame_index = frame_idx
            current_seq_len += 1  # account for the decision token

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                cache_position = torch.tensor([current_seq_len - 1], device=self.device)
                outputs = model_to_use(
                    input_ids=next_token,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                del outputs

            if next_token_id == self.frame_interval_token_id:
                if self.is_main_process():
                    print(f"  Decision: CONTINUE (,)")
                continue
                
            elif next_token_id == self.frame_end_token_id:
                if self.is_main_process():
                    print(f"  Decision: STOP (\\n)")
                generated_text, timing_stats = generate_final_response(past_key_values, current_seq_len)
                break
            
            else:
                if self.is_main_process():
                    print(f"  Decision: UNEXPECTED ('{self.tokenizer.decode([next_token_id])}'), treating as STOP")
                generated_text, timing_stats = generate_final_response(past_key_values, current_seq_len)
                break

        # Ensure generated_text and timing_stats are always defined.
        if 'generated_text' not in locals() or timing_stats is None:
            generated_text, timing_stats = generate_final_response(past_key_values, current_seq_len)

        if self.is_main_process():
            print(f"\n{'='*70}")
            print(f"Generation complete")
            print(f"{'='*70}")
            print(f"  Frames used after query: {current_frame_index + 1}/{len(images_after)}")
            print(f"  Generated text length: {len(generated_text)} chars")
            print(f"  Generated text preview: {generated_text[:200]}...")
            print(f"{'='*70}\n")
        
        del past_key_values, images_before, remaining_images_after
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        self._print_memory_stats("After final generation")
        
        return current_frame_index, generated_text, timing_stats
    
    def evaluate_dataset(self, data_path: str, image_root: str, output_path: str, max_samples: int = None):
        print(f"Loading dataset from {data_path}...")
        
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        
        if max_samples is not None:
            dataset = dataset[:max_samples]
        
        total_samples = len(dataset)
        if self.local_rank != -1:
            # Ceil-divide so every rank gets a contiguous, evenly-sized slice.
            samples_per_rank = (total_samples + self.world_size - 1) // self.world_size
            start_idx = self.local_rank * samples_per_rank
            end_idx = min(start_idx + samples_per_rank, total_samples)
            dataset = dataset[start_idx:end_idx]
            
            if self.is_main_process():
                print(f"Total samples: {total_samples}, samples per rank: ~{samples_per_rank}")
            print(f"Rank {self.local_rank}: processing samples {start_idx} to {end_idx} ({len(dataset)} samples)")

        
        results = []
        iterator = tqdm(dataset, desc=f"Rank {self.local_rank}") if self.is_main_process() else dataset
        
        for sample_idx, sample in enumerate(iterator):
            try:
                image_paths = sample.get('images', [])
                images = []
                for img_path in image_paths:
                    full_path = os.path.join(image_root, img_path)
                    if os.path.exists(full_path):
                        images.append(Image.open(full_path).convert('RGB'))
                
                conversations = sample.get('conversations', [])
                conversation_value = None
                ground_truth = None
                
                for conv in conversations:
                    if conv.get('from') == 'human' or conv.get('role') == 'user':
                        conversation_value = conv.get('value', '')
                    elif conv.get('from') == 'gpt' or conv.get('role') == 'assistant':
                        ground_truth = conv.get('value', '')
                
                if not images or not conversation_value:
                    # Add a placeholder so every rank keeps the same number of entries.
                    results.append({
                        'sample_id': sample_idx,
                        'skipped': True,
                        'reason': 'missing images or conversation'
                    })
                    continue
                
                images_before, query_text, images_after = self.parse_conversation_with_query(conversation_value)
                num_images_before = len(images_before)
                num_images_after = len(images_after)
                
                last_frame_idx, generated_text, timing_stats = self.generate_with_streaming_decision(
                    images, query_text, num_images_before
                )
                
                result = {
                    'sample_id': sample_idx,
                    'query': query_text,
                    'ground_truth': ground_truth,
                    'prediction': generated_text,
                    'question_time': sample.get('question_time', None),
                    'prediction_answer_time': float(num_images_before + last_frame_idx),
                    'gt_answer_time': sample.get('answer_for_test_time', None),
                    'test_type': sample.get('test_type', 'unknown'),
                    'question_type': sample.get('question_type', 'unknown'),
                    'num_images_before_query': num_images_before,
                    'num_images_after_query': num_images_after,
                    'last_frame_index_after_query': last_frame_idx,
                    'num_images_used_after_query': last_frame_idx + 1,
                    'total_images': len(images),
                    'timing': {
                        'ttft_seconds': timing_stats['ttft'],
                        'ttft_ms': timing_stats['ttft'] * 1000,
                        'answer_generation_latency_seconds': timing_stats['answer_generation_latency'],
                        'answer_generation_latency_ms': timing_stats['answer_generation_latency'] * 1000,
                        'num_output_tokens': timing_stats['num_output_tokens'],
                        'tokens_per_second': timing_stats['tokens_per_second'],
                    }
                }
                results.append(result)
            
            except Exception as e:
                print(f"\nError processing sample {sample_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Add a placeholder on error so every rank keeps the same number of entries.
                results.append({
                    'sample_id': sample_idx,
                    'skipped': True,
                    'reason': str(e)
                })
                continue

        # Synchronize all ranks before gathering results.
        if self.local_rank != -1:
            print(f"Rank {self.local_rank}: Finished processing {len(results)} samples, waiting for sync...")
            dist.barrier()

            try:
                all_results = [None] * self.world_size
                dist.all_gather_object(all_results, results)

                # Merge results from every rank and drop skipped placeholders.
                results = []
                for rank_results in all_results:
                    if rank_results:
                        for r in rank_results:
                            if not r.get('skipped', False):
                                results.append(r)

                print(f"Rank {self.local_rank}: Gathered {len(results)} valid results from all ranks")
            except Exception as e:
                print(f"Rank {self.local_rank}: all_gather_object failed: {e}")
                # Fall back to local results if gather fails.
                if self.is_main_process():
                    results = [r for r in results if not r.get('skipped', False)]

        # Save results only on the main process.
        if self.is_main_process():
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to {output_path}")
            self._print_statistics(results)
    
    def _print_statistics(self, results: List[Dict]):
        total = len(results)
        if total == 0:
            return
        
        frames_used = [r['num_images_used_after_query'] for r in results]
        frames_total = [r['num_images_after_query'] for r in results]

        ttft_values = [r['timing']['ttft_ms'] for r in results if 'timing' in r]
        latency_values = [r['timing']['answer_generation_latency_ms'] for r in results if 'timing' in r]
        tokens_per_sec_values = [r['timing']['tokens_per_second'] for r in results if 'timing' in r]
        output_tokens_values = [r['timing']['num_output_tokens'] for r in results if 'timing' in r]
        
        print("\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        print(f"Total samples: {total}")
        print(f"Average frames used: {sum(frames_used)/len(frames_used):.2f}")
        print(f"Average frames available: {sum(frames_total)/len(frames_total):.2f}")
        
        stopped_early = sum(1 for r in results if r['num_images_used_after_query'] < r['num_images_after_query'])
        print(f"Early stopping rate: {stopped_early/total:.1%}")
        
        if ttft_values:
            print(f"\n⏱️  TIMING STATISTICS:")
            print(f"  {'='*50}")
            print(f"  TTFT (Time to First Token):")
            print(f"    Average: {sum(ttft_values)/len(ttft_values):.2f}ms")
            print(f"    Min: {min(ttft_values):.2f}ms")
            print(f"    Max: {max(ttft_values):.2f}ms")
            
            print(f"\n  Answer Generation Latency:")
            print(f"    Average: {sum(latency_values)/len(latency_values):.2f}ms")
            print(f"    Min: {min(latency_values):.2f}ms")
            print(f"    Max: {max(latency_values):.2f}ms")
            
            print(f"\n  Token Generation:")
            print(f"    Average output tokens: {sum(output_tokens_values)/len(output_tokens_values):.1f}")
            print(f"    Average tokens/sec: {sum(tokens_per_sec_values)/len(tokens_per_sec_values):.2f}")
            print(f"  {'='*50}")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model's streaming capabilities with memory optimization")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--image_root", type=str, default="/Stream3D-Bench")
    parser.add_argument("--output_path", type=str, default="/output_logs/Stream3D-VLM-4B/evaluate_results.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                       help="Disable gradient checkpointing (use more memory but faster)")
    parser.add_argument(
        "--noisy_geometry", type=str, default="none",
        help="Replace the StreamVGGT geometry prior with noise for ablation. "
             "Supported specs: 'none' (default), 'zero', 'random', 'random_<std>', "
             "'matched' (Gaussian calibrated to real-feature mean/std), "
             "'perturb_<std>', 'shuffle'.",
    )
    parser.add_argument(
        "--noisy_geometry_seed", type=int, default=42,
        help="Seed for the noisy-prior patcher (reproducibility).",
    )
    parser.add_argument(
        "--swap_geometry_encoder", type=str, default="none",
        help="Replace the geometry encoder *at test time* with another "
             "reconstruction backbone for ablation. Supported: "
             "'none' (default, keep StreamVGGT from the checkpoint), "
             "'pi3', 'vggt', 'streamvggt', 'cut3r'. The trained merger / "
             "fusion modules are kept and consume the new encoder's features.",
    )
    parser.add_argument(
        "--swap_geometry_path", type=str, default=None,
        help="Pretrained-model path / HF id passed to the swapped encoder's "
             "load_model() (e.g. /path/to/Pi3 or facebook/VGGT-1B).",
    )
    parser.add_argument(
        "--swap_skip_feature_dim_check", action="store_true",
        help="By default we assert the swapped encoder's feature_dim matches "
             "the trained mergers (2048). Pass this flag to bypass that "
             "guard if you know what you are doing.",
    )

    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))

    world_size = 1
    if local_rank != -1:
        # Use a longer NCCL timeout for very long evaluations.
        import datetime
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=2)
        )
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_path = args.output_path.replace(".json", f"_{timestamp}.json")
    
    # Create evaluator
    evaluator = StreamingEvaluator(
        model_path=args.model_path,
        local_rank=local_rank,
        world_size=world_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    if args.swap_geometry_encoder and args.swap_geometry_encoder.lower() not in ("none", "off", ""):
        inner_model = (
            evaluator.model.module if hasattr(evaluator.model, "module") else evaluator.model
        )
        swap_geometry_encoder(
            inner_model,
            encoder_type=args.swap_geometry_encoder,
            model_path=args.swap_geometry_path,
            expected_feature_dim=None if args.swap_skip_feature_dim_check else 2048,
            verbose=evaluator.is_main_process(),
        )
        # Reflect the swap in the output filename so the JSON is self-describing.
        tag = args.swap_geometry_encoder.replace("/", "-")
        if "_swapenc-" not in args.output_path:
            args.output_path = args.output_path.replace(
                ".json", f"_swapenc-{tag}.json"
            )

    # Optional ablation: replace the geometry prior with noise.
    if args.noisy_geometry and args.noisy_geometry.lower() not in ("none", "off", ""):
        inner_model = (
            evaluator.model.module if hasattr(evaluator.model, "module") else evaluator.model
        )
        patch_geometry_encoder_with_noise(
            inner_model,
            spec=args.noisy_geometry,
            seed=args.noisy_geometry_seed + (local_rank if local_rank != -1 else 0),
            verbose=evaluator.is_main_process(),
        )
        # Reflect the spec in the output filename so the JSON is self-describing.
        tag = args.noisy_geometry.replace("/", "-")
        if "_noisygeo-" not in args.output_path:
            args.output_path = args.output_path.replace(
                ".json", f"_noisygeo-{tag}.json"
            )

    # Run evaluation
    evaluator.evaluate_dataset(
        data_path=args.data_path,
        image_root=args.image_root,
        output_path=args.output_path,
        max_samples=args.max_samples,
    )
    
    # Cleanup
    if local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()