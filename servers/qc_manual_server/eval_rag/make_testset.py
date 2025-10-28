import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from ragas.testset.persona import Persona
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

path = "/Users/xiaohuxu/Documents/python/sob_blobs/sobereva_blogs_text/about_qc/test"
loader = DirectoryLoader(path, glob="*.md")
docs = loader.load()
print(len(docs))

chatmodel =  ChatOpenAI(
                model="gpt-4o-mini",
                base_url="https://openai.weavex.tech/v1"
                )

chatmodel = ChatOpenAI(
    api_key="sk-f22c4fa77bab4a42a47486922c84a467",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
)

# embeddingmodel = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     openai_api_base="https://openai.weavex.tech/v1/"
#     )
embeddingmodel = DashScopeEmbeddings(model="text-embedding-v4",)

generator_llm = LangchainLLMWrapper(chatmodel)
generator_embeddings = LangchainEmbeddingsWrapper(embeddingmodel)

personas = [
    Persona(
        name="充满好奇心的学生",
        role_description="你是一个对计算化学和量子化学充满好奇心的学生，渴望了解更多关于这些领域的知识。",
    ),
]

# transforms = [HeadlineSplitter()]
transforms = [HeadlineSplitter(), NERExtractor()]

generator = TestsetGenerator(
    llm=generator_llm, 
    embedding_model=generator_embeddings, 
    persona_list=personas
)

distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
]

import asyncio

async def main():
    # for query, _ in distribution:
    #     prompts = await query.adapt_prompts("chinese", llm=generator_llm)
    #     query.set_prompts(**prompts)

    dataset = generator.generate_with_langchain_docs(
        docs[:],
        testset_size=5,
        transforms=transforms,
        query_distribution=distribution,
    )
    
    eval_dataset = dataset.to_evaluation_dataset()

    print("Query:", eval_dataset[0].user_input)
    print("Reference:", eval_dataset[0].reference)

if __name__ == "__main__":
    asyncio.run(main())
