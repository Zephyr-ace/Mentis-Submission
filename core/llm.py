from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class LLM_OA:
    def __init__(self, default_model: str):
        self.default_model = default_model
        self.client = OpenAI()  # nutzt automatisch environment variable "OPENAI_API_KEY"
        self.async_client = AsyncOpenAI()  # für parallele Verarbeitung

    async def __aenter__(self) -> "LLM_OA":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.async_client.close()

    def generate(self, prompt: str, model: str = None) -> str:
        model = model or self.default_model
        response = self.client.responses.create(
            model=model,
            input=[
                {"role": "user",
                 "content": prompt}
            ]
        )
        return response.output_text

    def generate_structured(self, prompt: str, desired_output_format: Type[T], model: str = None) -> T:
        model = model or self.default_model

        response = self.client.responses.parse(
            model=model,
            input=[{"role": "user", "content": prompt}],
            text_format=desired_output_format,
        )
        return response.output_parsed


    async def generate_parallel(self, prompts: list[str], model: str = None) -> list[str]:
        model = model or self.default_model

        async def async_generate(prompt: str) -> str:
            response = await self.async_client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}]
            )
            return response.output_text

        tasks = [async_generate(p) for p in prompts]
        return await asyncio.gather(*tasks)

# generate structured parallel methods
# <> NOT SURE IF EVEN NEEDED!

    async def generate_structured_async(self, prompt: str, desired_output_format: Type[T],
                                        model: str = None) -> T:
        model = model or self.default_model

        response = await self.async_client.responses.parse(
            model=model,
            input=[{"role": "user", "content": prompt}],
            text_format=desired_output_format,
        )
        return response.output_parsed


    async def generate_structured_parallel(self,  prompts: list[str], desired_output_format: Type[T]) -> list[T]:
        async def generate_with_client(prompt: str) -> T:
            response = await self.async_client.responses.parse(
                model=self.default_model,
                input=[{"role": "user", "content": prompt}],
                text_format=desired_output_format,
            )
            return response.output_parsed
        
        tasks = [generate_with_client(prompt) for prompt in prompts]
        async_tasks = await asyncio.gather(*tasks)
        return async_tasks

# <>

    def generate_structured_parallel_sync(self, prompts: list[str], desired_output_format: Type[T]) -> list[T]:
        async def run_with_new_client():
            async_client = AsyncOpenAI()
            try:
                async def generate_with_client(prompt: str) -> T:
                    response = await async_client.responses.parse(
                        model=self.default_model,
                        input=[{"role": "user", "content": prompt}],
                        text_format=desired_output_format,
                    )
                    return response.output_parsed
                
                tasks = [generate_with_client(prompt) for prompt in prompts]
                return await asyncio.gather(*tasks)
            finally:
                await async_client.close()
        
        return asyncio.run(run_with_new_client())

