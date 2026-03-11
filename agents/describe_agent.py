from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
import os
from dotenv import load_dotenv
import base64
import mimetypes

load_dotenv()

class DescribeState(TypedDict):
    messages: Annotated[list, add_messages]
    base64_img: str
    
class DescribeAgent:
    def __init__(self):
        self.model = ChatOpenAI(
            model="nvidia/nemotron-nano-12b-v2-vl:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            max_retries=2,
            max_tokens=8192,
            temperature=0.4,
        )
        
        self.graph = self._build_graph()
        
    def _build_graph(self):
        graph = StateGraph(DescribeState)
        
        graph.add_node("describe_content", self._describe_content)
        
        graph.add_edge(START, "describe_content")
        graph.add_edge("describe_content", END)
        
        return graph.compile()
        
    def _encode_image_to_data_url(self, image_path: str):
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"
            
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        return f"data:{mime_type};base64,{base64_image}"
    
    def _describe_content(self, state: DescribeState):
        system_prompt = SystemMessage(
            content="""Ты — продвинутый AI-аналитик визуальных данных (Vision Analyst).
Твоя цель — извлечь МАКСИМУМ полезной информации из изображения научной статьи (графика, диаграммы, таблицы или схемы).

### ЗАДАЧИ:
1. **Идентификация:** Что это? (График зависимости, схема архитектуры, таблица результатов?).
2. **Анализ данных:**
   - Если график: Какие оси? Какие тренды? Где пик? Где провал?
   - Если таблица: Какая модель/метод побеждает (жирным)? Насколько велик отрыв?
   - Если схема: Какие блоки связаны? В чем суть потока данных?
3. **Вывод:** Какую мысль авторы хотели донести этой картинкой? (Например: "Наш метод X быстрее Y в 2 раза").

Отвечай на **русском языке**. Используй четкую структуру и маркированные списки."""
        )
        base64_img_data = self._encode_image_to_data_url(state['base64_img'])

        user_prompt = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Опиши содержимое изображения на русском языке:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_img_data
                    }
                }
            ]
        )

        response = self.model.invoke([system_prompt, user_prompt])
        return {
            "messages": [response]
        }
        
    def run(self, base64_img: str):
        initial_state = {
            "messages": [],
            "base64_img": base64_img
        }
        response = self.graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": "describe-agent-2"}}
        )
        return response['messages'][-1].content
    
    def run_with_state(self, desc_state: dict):
        response = self.graph.invoke(
            desc_state
        )
        
        return response
    
if __name__ == "__main__":
    agent = DescribeAgent()
    description = agent.run(base64_img="/Users/switchblade/Documents/vs_code/science_helpy_3/downloads/graph_mas_2026-03-05.png")
    print(description)
