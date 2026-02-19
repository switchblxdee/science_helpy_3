import os
import re
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Annotated, TypedDict, Optional, List, Literal
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from datetime import date
import io
from PIL import Image

from agents.describe_agent import DescribeAgent
from agents.review_agent import EvalAgent
from agents.writer_agent import WriterAgent
from agents.coordinator_agent import CoordinatorAgent

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

class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    selected_paper_path: Optional[str]
    paper_content: Optional[str]
    base64_img: Optional[str]
    image_description: Optional[str]
    review_data: Optional[PaperReview]
    written_review: Optional[str]
    next_node: Literal["coordinator", "describe", "eval", "writer", "end"]

class GraphMAS:
    def __init__(self):
        self.date = date.today().isoformat()
        
        self.coordinator_agent = CoordinatorAgent()
        self.describe_agent = DescribeAgent()
        self.review_agent = EvalAgent()
        self.writer_agent = WriterAgent()

        self.memory = MemorySaver()
        
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(MainState)

        graph.add_node("coordinator_agent", self._run_coordinator_agent)
        graph.add_node("eval_agent", self._run_review_agent)
        graph.add_node("writer_agent", self._run_write_agent)
        
        graph.add_edge(START, "coordinator_agent")
        graph.add_conditional_edges(
            "coordinator_agent",
            self._router,
            {
                "coordinator_agent": "coordinator_agent",
                "eval": "eval_agent",
                "writer": "writer_agent",
                "end": END
            }
        )
        graph.add_edge("eval_agent", "coordinator_agent")
        graph.add_edge("writer_agent", END)
        
        return graph.compile(checkpointer=self.memory)
       
    
    def _router(self, state: MainState) -> str:
        last_message = state["messages"][-1]
        content = last_message.content.upper()
        
        if "[EVAL]" in content:
            destination = "eval"
        # elif "[DESCRIBE]" in content:
        #     destination = "describe"
        elif "[WRITE]" in content:
            destination = "writer"
        elif "[END]" in content:
            destination = "end"
        else:
            destination = "end"

        print(f"[graph] router --> {destination}")
        return destination
        

    def _extract_tool_results(self, messages: list) -> dict:
        paper_content = None
        selected_paper_path = None

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            tool_name = getattr(msg, "name", None)

            if tool_name == "parse_pdf_file" and msg.content:
                paper_content = msg.content

            elif tool_name == "download_arxiv_paper" and msg.content:
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and data.get("status") == "success":
                        selected_paper_path = data.get("path")
                except (json.JSONDecodeError, TypeError):
                    match = re.search(r'Путь:\s*(\S+\.pdf)', msg.content)
                    if match:
                        selected_paper_path = match.group(1)

        return {"paper_content": paper_content, "selected_paper_path": selected_paper_path}

    _COORD_MESSAGE_WINDOW = 10

    def _build_coord_context(self, state: MainState) -> list:
        paper_content = state.get("paper_content")
        selected_paper_path = state.get("selected_paper_path")
        review_data = state.get("review_data")
        written_review = state.get("written_review")

        recent_messages = list(state["messages"])[-self._COORD_MESSAGE_WINDOW:]

        if not (paper_content and selected_paper_path):
            return recent_messages

        done_parts = [f"Статья загружена: {selected_paper_path}"]
        if review_data:
            pros = "; ".join(review_data.pros)
            cons = "; ".join(review_data.cons)
            done_parts.append(
                f"Оценка:\n"
                f"  Итог: {review_data.scores.overall}/5 | "
                f"Новизна: {review_data.scores.novelty} | "
                f"Строгость: {review_data.scores.rigor} | "
                f"Влияние: {review_data.scores.impact}\n"
                f"  Резюме: {review_data.one_sentence_summary}\n"
                f"  Сильные: {pros}\n"
                f"  Слабые: {cons}\n"
                f"  Обоснование: {review_data.reasoning}"
            )
        if written_review:
            done_parts.append("Обзор статьи уже написан и показан пользователю.")

        instruction = (
            "\nНЕ вызывай parse_pdf_file и download_arxiv_paper повторно. "
            "Если получена оценка (EVAL) или описание (DESCRIBE) — ТЫ ОБЯЗАН включить этот текст в свой финальный ответ, "
            "чтобы пользователь его увидел. Не скрывай результат. "
            "Когда ответ готов — заверши тегом [END]."
        )

        context_hint = HumanMessage(
            content=(
                "[SYSTEM CONTEXT] Статья уже загружена и обработана.\n"
                + "\n".join(f"- {p}" for p in done_parts)
                + instruction
            )
        )
        return [context_hint] + recent_messages

    def _run_coordinator_agent(self, state: MainState) -> MainState:
        paper_content = state.get("paper_content")
        selected_paper_path = state.get("selected_paper_path")

        messages_for_coord = self._build_coord_context(state)
        coord_state = {"messages": messages_for_coord}

        result = self.coordinator_agent.run_with_state(coord_state)

        extracted = self._extract_tool_results(result["messages"])
        new_paper_content = extracted["paper_content"] or paper_content
        new_paper_path = extracted["selected_paper_path"] or selected_paper_path

        if extracted["paper_content"]:
            print(f"[graph] paper_content сохранён ({len(extracted['paper_content'])} символов)")
        if extracted["selected_paper_path"]:
            print(f"[graph] selected_paper_path: {extracted['selected_paper_path']}")

        final_message = result["messages"][-1]

        return {
            **state,
            "messages": [final_message],
            "paper_content": new_paper_content,
            "selected_paper_path": new_paper_path,
        }

    def _run_write_agent(self, state: MainState) -> MainState:
        paper_text = state.get("paper_content") or state["messages"][-1].content
        print(f"[graph] --> WriterAgent (текст: {len(paper_text)} символов)")

        write_state = {"messages": [HumanMessage(content=paper_text)]}
        result = self.writer_agent.run_with_state(write_state)

        final_message = result["messages"][-1]
        print(f"[graph] <-- WriterAgent завершил работу")

        return {
            **state,
            "messages": [final_message],
            "written_review": final_message.content,
        }

    def _run_describe_agent(self, state: MainState) -> MainState:
        img_path = state.get("base64_img")
        print(f"[graph] --> DescribeAgent (изображение: {img_path})")

        describe_state = {
            "messages": [HumanMessage(content="Опиши это изображение")],
            "base64_img": img_path,
        }
        result = self.describe_agent.run_with_state(describe_state)

        final_message = result["messages"][-1]
        print(f"[graph] <-- DescribeAgent завершил работу")

        return {
            **state,
            "messages": [final_message],
            "image_description": final_message.content,
        }

    @staticmethod
    def _format_review(review_obj: PaperReview) -> str:
        pros = "\n".join(f"  + {p}" for p in review_obj.pros)
        cons = "\n".join(f"  - {c}" for c in review_obj.cons)
        return (
            f"РЕЗУЛЬТАТ ОЦЕНКИ СТАТЬИ\n\n"
            f"Категория NLP: {review_obj.nlp_category}\n"
            f"Релевантность: {'да' if review_obj.is_relevant else 'нет'}\n\n"
            f"Краткое резюме: {review_obj.one_sentence_summary}\n\n"
            f"Оценки:\n"
            f"  Новизна:              {review_obj.scores.novelty}/5\n"
            f"  Методологическая строгость: {review_obj.scores.rigor}/5\n"
            f"  Потенциальное влияние: {review_obj.scores.impact}/5\n"
            f"  ИТОГОВАЯ ОЦЕНКА:   {review_obj.scores.overall}/5\n\n"
            f"Сильные стороны:\n{pros}\n\n"
            f"Слабые стороны:\n{cons}\n\n"
            f"Обоснование: {review_obj.reasoning}"
        )

    def _run_review_agent(self, state: MainState) -> MainState:
        paper_to_eval = state.get("paper_content") or state["messages"][-1].content
        print(f"[graph] --> EvalAgent (текст: {len(paper_to_eval)} символов)")

        review_state = {"messages": paper_to_eval}
        result = self.review_agent.run_with_state(review_state)

        review_obj = result.get("review")
        print(f"[graph] <-- EvalAgent завершил работу (оценка: {review_obj.scores.overall if review_obj else 'ошибка'})")

        if review_obj:
            review_text = self._format_review(review_obj)
        else:
            review_text = "Ошибка: не удалось получить оценку статьи."

        return {
            **state,
            "messages": [HumanMessage(content=review_text)],
            "review_data": review_obj,
        }
        
    
    def run(self, user_query: str):
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
        }
        png_bytes = self.graph.get_graph().draw_mermaid_png()
        output_path = f"./graph_mas_{self.date}.png"
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"Граф сохранен в {output_path}")
        result = self.graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"graph-mas-{self.date}"}}
        )
        
        return result
    
if __name__ == "__main__":
    mas = GraphMAS()
    
    user_input = input("Здравствуйте! Введите запрос для научного помощника (или 'exit' для выхода):\n")
    while user_input.lower() != "exit":
        result = mas.run(user_input)
        
        last_msg = result["messages"][-1]
        print(f"\nОТВЕТ:\n{last_msg.content}\n")
        
        if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
            usage = last_msg.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            print(f"--- Статистика токенов ---")
            print(f"Вход (Prompt): {input_tokens}")
            print(f"Выход (Completion): {output_tokens}")
            print(f"Всего: {total_tokens}")
            print(f"--------------------------")
        
        user_input = input("Введите запрос: ")
