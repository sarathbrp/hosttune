from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from tune.infrastructure.model_config import ModelEndpointConfig


class HypothesisGraphState(TypedDict):
    prompt: str
    response: str


@dataclass
class LangGraphHypothesisClient:
    config: ModelEndpointConfig
    _graph: object | None = field(default=None, init=False, repr=False)

    def complete(self, prompt: str) -> str:
        graph = self._get_graph()
        result = graph.invoke({"prompt": prompt, "response": ""})
        response = result.get("response")
        if not isinstance(response, str) or response == "":
            msg = "LangGraph hypothesis client returned an empty response."
            raise ValueError(msg)
        return response

    def _get_graph(self) -> object:
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> object:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as error:
            msg = "langgraph is required for LangGraphHypothesisClient."
            raise RuntimeError(msg) from error

        graph = StateGraph(HypothesisGraphState)
        graph.add_node("call_model", self._call_model_node)
        graph.add_edge(START, "call_model")
        graph.add_edge("call_model", END)
        return graph.compile()

    def _call_model_node(self, state: HypothesisGraphState) -> HypothesisGraphState:
        prompt = state["prompt"]
        client = self._build_openai_client()
        completion = client.chat.completions.create(
            model=self.config.model_name,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce one tuning hypothesis in strict JSON. "
                        "Do not invent unsupported parameter keys."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content
        if not isinstance(content, str):
            msg = "OpenAI-compatible response did not include string content."
            raise ValueError(msg)
        return {"prompt": prompt, "response": content}

    def _build_openai_client(self) -> object:
        try:
            from openai import OpenAI
        except ImportError as error:
            msg = "openai is required for LangGraphHypothesisClient."
            raise RuntimeError(msg) from error
        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
