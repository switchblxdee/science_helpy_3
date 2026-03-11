import os
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class PaperScores(BaseModel):
    novelty: int = Field(..., description="Score 1-5 for novelty")
    rigor: int = Field(..., description="Score 1-5 for methodological rigor")
    impact: int = Field(..., description="Score 1-5 for potential impact")
    overall: int = Field(..., description="Weighted overall score 1-5")

class PaperReview(BaseModel):
    nlp_category: str = Field(..., description="Core NLP, Multimodal, Tangential, or Not NLP")
    is_relevant: bool = Field(..., description="True if category is Core NLP or Multimodal")
    one_sentence_summary: str = Field(..., description="Concise summary of contribution")
    scores: PaperScores
    pros: List[str] = Field(..., description="List of paper's strengths")
    cons: List[str] = Field(..., description="List of paper's weaknesses")
    reasoning: str = Field(..., description="Justification for the scores")


class EvalState(TypedDict):
    messages: str
    review: Optional[PaperReview]


class EvalAgent:
    def __init__(self):
        self.model = ChatOpenAI(
            model="arcee-ai/trinity-large-preview:free", 
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            temperature=0.1,
            max_tokens=9128,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(EvalState)

        graph.add_node("evaluate_node", self._evaluate_paper_node)
        
        graph.add_edge(START, "evaluate_node")
        graph.add_edge("evaluate_node", END)

        return graph.compile()

    def _evaluate_paper_node(self, state: EvalState) -> EvalState:

        system_text = """Ты — старший рецензент (Senior Reviewer) ведущих конференций по ИИ (ACL, NeurIPS, ICLR).
Твоя задача — дать глубокую, критическую и объективную оценку научной статьи.

### ВАЖНО:
Входные данные содержат:
1. **Описания изображений** (графики, схемы, таблицы) — помечены как "=== ОПИСАНИЯ ИЗОБРАЖЕНИЙ ===" (в начале)
2. **Текст статьи** (из LaTeX .tex файлов или PDF) — помечен как "=== ТЕКСТ СТАТЬИ ==="

Используй описания изображений для оценки:
- Если в статье есть графики/таблицы — оцени, насколько они информативны и подтверждают ли выводы.
- Если есть схемы архитектуры — оцени, насколько методология ясна.
- Отсутствие изображений или их слабое качество — минус в пользу методологии.

### КРИТЕРИИ ОЦЕНКИ (СТРОГО):

1. **Новизна (1-5):**
   - 1: Идея тривиальна или уже известна.
   - 3: Небольшое улучшение существующего метода.
   - 5: Прорыв, новая парадигма (как Transformer или Diffusion Models).

2. **Методология (1-5):**
   - 1: Эксперименты некорректны, сравнения слабые.
   - 3: Стандартные эксперименты, есть небольшие вопросы.
   - 5: Безупречная методология, обширные абляции, воспроизводимость.

3. **Влияние (1-5):**
   - 1: Никому не интересно.
   - 3: Полезно для узкой ниши.
   - 5: Станет базовой статьей (citation classic).

### ФОРМАТ ОТВЕТА (JSON):
Ты должен вернуть результат СТРОГО в формате JSON, соответствующем схеме `PaperReview`.
В поле `reasoning` дай развернутое обоснование (минимум 3-4 предложения на каждый критерий). Будь жестким, но справедливым. Не используй общие фразы."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            ("human", "Вот содержимое/аннотация статьи:\n\n{paper_text}")
        ])
        
        structured_llm = self.model.with_structured_output(PaperReview)
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({"paper_text": state["messages"]})
            return {"review": result}
        except Exception as e:
            print(result)
            print(f"Error parsing model output: {e}")
            return {"review": None}

    def evaluate(self, paper_text: str) -> dict:
        result = self.graph.invoke({
            "paper": paper_text,
            "review": None
        })
        
        if result["review"]:
            return result["review"].model_dump()
        else:
            return {"error": "Failed to evaluate paper"}
        
    def run_with_state(self, review_state: dict):
        
        result = self.graph.invoke(
            review_state
        )
        
        return result

if __name__ == "__main__":
    agent = EvalAgent()
    
    weak_paper = """
    We propose a method to classify sentiment using a simple bag-of-words model on the IMDB dataset.
    Our results show 85% accuracy, which is good but not state of the art. 
    We just ran the code from a tutorial.
    """
    
    strong_paper = """
    Title: Attention Is All You Need.
    We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, 
    dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks 
    show these models to be superior in quality while being more parallelizable and requiring significantly 
    less time to train.
    """

    print("--- Evaluating Weak Paper ---")
    res1 = agent.evaluate(weak_paper)
    import json
    print(json.dumps(res1, indent=2))

    print("\n--- Evaluating Strong Paper ---")
    res2 = agent.evaluate(strong_paper)
    print(json.dumps(res2, indent=2))
