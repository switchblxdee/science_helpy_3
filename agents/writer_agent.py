from typing import Annotated, TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
import os

load_dotenv()

class WriterState(TypedDict):
    messages: Annotated[list, add_messages]
     
class WriterAgent:
    def __init__(self):
        self.tools = [TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))]
        self.model = ChatOpenAI(
            model="arcee-ai/trinity-large-preview:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            max_retries=2,
            max_tokens=9128,
            temperature=0.7,
        ).bind_tools(self.tools)
        
        self.memory = MemorySaver()
        self.tool_node = ToolNode(self.tools)
        
        self.graph = self._build_graph(WriterState) 
        
    def _build_graph(self, state: WriterState):
        graph = StateGraph(state)
        
        graph.add_node("write_review", self._write_review)
        graph.add_node("tools", self.tool_node)
        
        graph.add_edge(START, "write_review")
        graph.add_conditional_edges(
            "write_review",
            self._should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        graph.add_edge("tools", "write_review")
        
        return graph.compile(checkpointer=self.memory)

    def _write_review(self, state: WriterState) -> WriterState:
        system_prompt = SystemMessage(
            content="""Ты — экспертный технический писатель и аналитик в области ИИ.
Твоя задача — создать глубокий, структурированный и увлекательный обзор научной статьи для профессионального сообщества (ученых, инженеров ML).

### ТВОЯ МИССИЯ:
- Превратить сложный научный текст в понятный, но не упрощенный обзор.
- Выделить самое главное ("соль"), отбросив воду.
- Дать критическую оценку, а не просто пересказать аннотацию.

### СТРУКТУРА ОБЗОРА (ОБЯЗАТЕЛЬНО):

1. **🚀 Краткое резюме (Executive Summary):**
   - О чем статья в одном абзаце? Какую проблему решает? Какой главный вклад (contribution)?

2. **🔑 Ключевые идеи и методы:**
   - Как именно работает предложенный метод? (Используй термины, но объясняй их суть).
   - В чем отличие от SOTA (предыдущих лучших решений)?

3. **📊 Результаты и эксперименты:**
   - На каких бенчмарках тестировали?
   - Какие метрики улучшились и на сколько? (Цифры важны!).

4. **⚖️ Сильные и слабые стороны (Critique):**
   - Плюсы: Что авторы сделали круто?
   - Минусы: Чего не хватает? Где метод может сломаться? Были ли честные абляции?

5. **💡 Практическое применение:**
   - Кому и зачем это нужно прямо сейчас? Где это можно внедрить?

6. **🏁 Вердикт:**
   - Стоит ли читать фулл? (Мастхэв / Проходная / Интересно только узким спецам).

### СТИЛЬ:
- Язык: **Русский** (профессиональный, живой).
- Формат: Markdown (жирные заголовки, списки, выделение ключевых мыслей).
- Тон: Экспертный, объективный, слегка критичный.
- Объем: Достаточный для понимания сути без чтения оригинала (около 500-800 слов).

Используй поиск (Tavily), если встречаешь неизвестные термины или хочешь найти контекст (кто авторы, где опубликовано)."""
        )

        last_message = state['messages'][-1]
        article_text = last_message.content if hasattr(last_message, 'content') else str(last_message)

        user_prompt = HumanMessage(
            content=f"""Напиши обзор на основе следующего текста статьи:\n\n{article_text}"""
        )
        
        response = self.model.invoke([system_prompt, user_prompt])
        
        return {
            "messages": [response]
        }
        
    def _should_continue(self, state: WriterState) -> str:
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("used tool")
            return "tools"
        
        return "end"
    
    def run(self, paper: str):
        initial_state = {
            "messages": [HumanMessage(content=paper)]
        }

        response = self.graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": "writer-agent-1"}}
        )
        return response["messages"][-1].content
    
    def run_with_state(self, write_state: dict):
        result = self.graph.invoke(
            write_state
        )
        
        return result
    
if __name__ == "__main__":
    agent = WriterAgent()
    sample_paper = """Трипадыпасыка"""
    review = agent.run(sample_paper)
    print(review)
