import json
import logging
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("vllm", "ollama")
OLLAMA_STRUCTURED_MAX_TOKENS = 4096


class LLMEngineManager:
    """
    Quản lý mô hình LLM chính, có 2 backend chọn bằng LLM_BACKEND:
      - "vllm": vLLM engine trên GPU NVIDIA (môi trường chạy thật, 4 GPU A16),
        ép cấu trúc đầu ra bằng Outlines cho Tool Calling (Quiz / Flashcard).
      - "ollama": gọi Ollama qua HTTP — dùng cho máy dev không có GPU NVIDIA
        (vLLM không chạy được trên GPU tích hợp AMD / Windows).
    """

    def __init__(self):
        self.backend = settings.llm_backend.strip().lower()
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"LLM_BACKEND='{settings.llm_backend}' is invalid, choices: {', '.join(SUPPORTED_BACKENDS)}."
            )

        self.llm: Optional[Any] = None
        self.outlines_model = None
        self._sampling_params_cls = None

        if self.backend == "ollama":
            self.ollama_base_url = settings.ollama_base_url.rstrip("/")
            self.ollama_model = settings.ollama_model_name
            logger.info(f"LLM backend = ollama ({self.ollama_model} @ {self.ollama_base_url})")
            return

        self.model_name = (
            settings.llm_model_name
            if settings.llm_model_name and settings.llm_model_name != "change-me"
            else "Qwen/Qwen2.5-7B-Instruct"
        )

        self.tensor_parallel_size = settings.llm_tensor_parallel_size

        # Tỷ lệ giới hạn VRAM sử dụng (0.85 = 85% VRAM mỗi GPU để tránh OOM)
        self.gpu_memory_utilization = settings.gpu_memory_utilization

        self._init_engine()

    def _init_engine(self):
        try:
            import torch
            from outlines import models
            from vllm import LLM, SamplingParams

            logger.info(f"Loading LLM Model '{self.model_name}' on {self.tensor_parallel_size} GPU...")

            if not torch.cuda.is_available():
                logger.warning("CUDA not available! vLLM requires environment with GPU.")
                return

            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=4096,  # Limit context window for RAG
                trust_remote_code=True
            )
            self._sampling_params_cls = SamplingParams

            self.outlines_model = models.vllm(self.llm)
            logger.info("Initialized vLLM Engine successfully.")

        except Exception as e:
            logger.error(f"Error initializing vLLM Engine: {str(e)}")

    def _ollama_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": settings.ollama_num_ctx,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if response_format is not None:
            payload["format"] = response_format

        response = httpx.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=settings.ollama_timeout_s,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream_rag_response(self, prompt, system_prompt, temperature=0.2, max_tokens=512):
        """Yield {"token": str} continuously, ending with {"done": True, ...statistics}."""
        if self.backend != "ollama":
            yield {"token": self.generate_rag_response(prompt, system_prompt, temperature, max_tokens)}
            yield {"done": True}
            return

        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": temperature,
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": max_tokens,
            },
        }
        with httpx.stream("POST", f"{self.ollama_base_url}/api/chat", json=payload, timeout=settings.ollama_timeout_s) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line: continue
                data = json.loads(line)
                if data.get("done"):
                    yield {"done": True,
                        "eval_count": data.get("eval_count"),
                        "eval_duration": data.get("eval_duration"),
                        "prompt_eval_duration": data.get("prompt_eval_duration")}
                    return
                token = data.get("message", {}).get("content", "")
                if token:
                    yield {"token": token}

    def _ollama_error_message(self, exc: Exception) -> str:
        if isinstance(exc, httpx.ConnectError):
            return (
                f"Error connecting to Ollama at {self.ollama_base_url}. "
                "Please check if Ollama is running on the server."
            )
        if isinstance(exc, httpx.TimeoutException):
            return "Ollama is responding too slowly (timeout). Try a smaller model or increase OLLAMA_TIMEOUT_S."
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
            return f"Do not have model '{self.ollama_model}' in Ollama. Please run: ollama pull {self.ollama_model}"
        return "An error occurred during model inference."

    def generate_rag_response(
        self, 
        prompt: str, 
        system_prompt: str, 
        temperature: float = 0.2, 
        max_tokens: int = 512
    ) -> str:
        if self.backend == "ollama":
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                return self._ollama_chat(messages, temperature, max_tokens=max_tokens).strip()
            except Exception as e:
                logger.error(f"Error with Ollama when generating RAG response: {str(e)}")
                return self._ollama_error_message(e)

        if not self.llm:
            return "AI Engine is not ready (Missing GPU/vLLM)."

        try:
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            sampling_params = self._sampling_params_cls(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|endoftext|>"]
            )

            outputs = self.llm.generate([full_prompt], sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            return response_text

        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return "An error occurred during model inference."

    def generate_structured_output(
        self, 
        prompt: str, 
        schema_class: type[BaseModel], 
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Thực thi Tool Calling: Ép 100% LLM trả về đúng định dạng JSON theo Pydantic Schema.
        vLLM: dùng Outlines. Ollama: truyền JSON schema vào tham số `format`,
        rồi validate lại bằng Pydantic (model nhỏ vẫn có thể sinh sai nội dung).
        
        :param prompt: Prompt mô tả câu hỏi / bài kiểm tra cần sinh
        :param schema_class: Lớp Pydantic định nghĩa JSON Schema (Quiz Schema, Flashcard Schema)
        """
        if self.backend == "ollama":
            schema = schema_class.model_json_schema()
            grounded_prompt = (
                f"{prompt}\n\nChỉ trả về JSON hợp lệ đúng theo schema sau:\n"
                f"{json.dumps(schema, ensure_ascii=False)}"
            )
            try:
                content = self._ollama_chat(
                    [{"role": "user", "content": grounded_prompt}],
                    temperature,
                    max_tokens=OLLAMA_STRUCTURED_MAX_TOKENS,
                    response_format=schema,
                )
            except Exception as e:
                logger.error(f"Error with Ollama when generating structured output: {str(e)}")
                raise RuntimeError(self._ollama_error_message(e)) from e

            try:
                return schema_class.model_validate_json(content).model_dump()
            except ValidationError as e:
                logger.error(f"Ollama returned JSON that does not match the schema: {str(e)}")
                raise ValueError("The model generated JSON does not match the expected structure.") from e

        if not self.outlines_model:
            raise RuntimeError("vLLM/Outlines Engine has not been initialized successfully.")

        try:
            from outlines import generate

            generator = generate.json(self.outlines_model, schema_class)
            result_model = generator(prompt, temperature=temperature)

            return result_model.model_dump()

        except JSONDecodeError as e:
            logger.error(f"JSON Decode Error when Tool Calling: {str(e)}")
            raise ValueError("Model generated JSON does not match the expected structure.") from e
        except Exception as e:
            logger.error(f"Critical Error when executing Structured Output: {str(e)}")
            raise e


# Singleton Instance để dùng chung toàn bộ ứng dụng
llm_engine = LLMEngineManager()
