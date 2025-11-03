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
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import (
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor,
)

path = "/Users/xiaohuxu/Documents/python/sob_blobs/sobereva_blogs_text/about_qc/test"
loader = DirectoryLoader(path, glob="*.md")
docs = loader.load()
print(len(docs))

chatmodel =  ChatOpenAI(
                model="gpt-4o-mini",
                base_url="https://openai.weavex.tech/v1"
                )

# chatmodel = ChatOpenAI(
#     api_key="sk-f22c4fa77bab4a42a47486922c84a467",
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     model="qwen-plus",
# )

embeddingmodel = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://openai.weavex.tech/v1/"
    )
# embeddingmodel = DashScopeEmbeddings(model="text-embedding-v4",)

generator_llm = LangchainLLMWrapper(chatmodel)
generator_embeddings = LangchainEmbeddingsWrapper(embeddingmodel)




kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )

llm = generator_llm
embeddings = generator_embeddings




headline_extractor = HeadlinesExtractor(llm=llm)
headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
keyphrase_extractor = KeyphrasesExtractor(
    llm=llm, property_name="keyphrases", max_num=10
)
transforms = [
    headline_extractor,
    headline_splitter,
    keyphrase_extractor,
]
apply_transforms(kg, transforms=transforms)

persona_list = [
    Persona(
        name="curious student",
        role_description="A student who is curious about the computational chemistry field and wants to learn more about it.",
    ),
]

from ragas.testset.synthesizers.single_hop import (
    SingleHopQuerySynthesizer,
    SingleHopScenario,
)
from dataclasses import dataclass
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)


@dataclass
class MySingleHopScenario(SingleHopQuerySynthesizer):

    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(self, n, knowledge_graph, persona_list, callbacks):

        property_name = "keyphrases"
        nodes = []
        for node in knowledge_graph.nodes:
            if node.type.name == "CHUNK" and node.get_property(property_name):
                nodes.append(node)

        number_of_samples_per_node = max(1, n // len(nodes))

        scenarios = []
        for node in nodes:
            if len(scenarios) >= n:
                break
            themes = node.properties.get(property_name, [""])
            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            base_scenarios = self.prepare_combinations(
                node,
                themes,
                personas=persona_list,
                persona_concepts=persona_concepts.mapping,
            )
            scenarios.extend(
                self.sample_combinations(base_scenarios, number_of_samples_per_node)
            )

        return scenarios

# query = MySingleHopScenario(llm=llm)

# scenarios = await query.generate_scenarios(
#     n=5, knowledge_graph=kg, persona_list=persona_list
# )

# scenarios[0]
import asyncio
async def main():
    query = MySingleHopScenario(llm=llm)

    scenarios = await query.generate_scenarios(
        n=5, knowledge_graph=kg, persona_list=persona_list
    )

    for scenario in scenarios:
        print(scenario)

asyncio.run(main())