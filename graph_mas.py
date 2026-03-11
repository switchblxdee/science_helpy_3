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
from pathlib import Path

from agents.describe_agent import DescribeAgent
from agents.review_agent import EvalAgent
from agents.writer_agent import WriterAgent
from agents.coordinator_agent import CoordinatorAgent
from agent_tools.tools import list_tex_images, parse_tex_file

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
    all_image_descriptions: Optional[List[str]]
    review_data: Optional[PaperReview]
    written_review: Optional[str]
    next_node: Literal["coordinator", "describe", "eval", "writer", "end"]
    extracted_images_path: Optional[str]

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
        graph.add_node("describe_images_for_eval", self._describe_images_for_eval)
        graph.add_node("describe_images_for_write", self._describe_images_for_write)

        graph.add_edge(START, "coordinator_agent")
        graph.add_conditional_edges(
            "coordinator_agent",
            self._router,
            {
                "coordinator_agent": "coordinator_agent",
                "eval": "describe_images_for_eval",
                "writer": "describe_images_for_write",
                "end": END
            }
        )
        graph.add_edge("describe_images_for_eval", "eval_agent")
        graph.add_edge("describe_images_for_write", "writer_agent")
        graph.add_edge("eval_agent", "coordinator_agent")
        graph.add_edge("writer_agent", END)

        return graph.compile(checkpointer=self.memory)
       
    
    def _router(self, state: MainState) -> str:
        last_message = state["messages"][-1]
        content = last_message.content.upper()

        if "[EVAL]" in content:
            destination = "eval"
        elif "[DESCRIBE]" in content:
            destination = "describe"
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
        extracted_images_path = None

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            tool_name = getattr(msg, "name", None)

            if tool_name in ("parse_pdf_file", "parse_tex_file") and msg.content:
                paper_content = msg.content

            elif tool_name == "parse_img_from_pdf" and msg.content:
                # Extract folder path from parse_img_from_pdf result
                # Format: "Successfully extracted X image(s) to:\n/path/to/folder/img1.png\n/path/to/folder/img2.png"
                if "successfully extracted" in msg.content.lower():
                    # Get the folder path from the first image path
                    lines = msg.content.split("\n")
                    for line in lines:
                        if "/extracted_images/" in line and line.strip().endswith(".png"):
                            # Extract folder path
                            img_path = line.strip()
                            extracted_images_path = str(Path(img_path).parent)
                            break

            elif tool_name == "download_arxiv_paper" and msg.content:
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, dict) and data.get("status") == "success":
                        selected_paper_path = data.get("path")
                except (json.JSONDecodeError, TypeError):
                    match = re.search(r'Путь:\s*(\S+\.pdf)', msg.content)
                    if match:
                        selected_paper_path = match.group(1)

        return {
            "paper_content": paper_content,
            "selected_paper_path": selected_paper_path,
            "extracted_images_path": extracted_images_path
        }

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
        extracted_images_path = state.get("extracted_images_path")

        messages_for_coord = self._build_coord_context(state)
        coord_state = {"messages": messages_for_coord}

        result = self.coordinator_agent.run_with_state(coord_state)

        extracted = self._extract_tool_results(result["messages"])
        new_paper_content = extracted["paper_content"] or paper_content
        new_paper_path = extracted["selected_paper_path"] or selected_paper_path
        new_extracted_images_path = extracted.get("extracted_images_path") or extracted_images_path

        if extracted["paper_content"]:
            print(f"[graph] paper_content сохранён ({len(extracted['paper_content'])} символов)")
        if extracted["selected_paper_path"]:
            print(f"[graph] selected_paper_path: {extracted['selected_paper_path']}")
        if extracted.get("extracted_images_path"):
            print(f"[graph] extracted_images_path: {extracted['extracted_images_path']}")

        final_message = result["messages"][-1]

        return {
            **state,
            "messages": [final_message],
            "paper_content": new_paper_content,
            "selected_paper_path": new_paper_path,
            "extracted_images_path": new_extracted_images_path,
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

    def _describe_images_for_eval(self, state: MainState) -> MainState:
        """
        Найти и описать все изображения в статье перед оценкой.
        """
        tex_path = state.get("selected_paper_path")
        extracted_images_path = state.get("extracted_images_path")
        
        if not tex_path and not extracted_images_path:
            return {
                **state,
                "messages": [HumanMessage(content="Ошибка: путь к статье не найден.")],
                "all_image_descriptions": [],
            }

        image_descriptions = []
        image_paths = []

        if extracted_images_path:
            print(f"[graph] --> Использование изображений из PDF: {extracted_images_path}")
            img_folder = Path(extracted_images_path).parent
            if img_folder.exists():
                for img_ext in ["*.png", "*.jpg", "*.jpeg"]:
                    image_paths.extend(list(img_folder.glob(img_ext)))
                image_paths = [str(p) for p in image_paths]
                print(f"[graph] Найдено изображений в PDF folder: {len(image_paths)}")

        if not image_paths and tex_path:
            if "_tex" not in tex_path:
                tex_path = tex_path.replace(".pdf", "_tex")

            print(f"[graph] --> Поиск изображений для EVAL в {tex_path}")

            images_result = list_tex_images.invoke({"tex_path": tex_path})
            print(f"[graph] {images_result[:200]}")

            if "Изображения не найдены" not in images_result and "Ошибка" not in images_result:
                for line in images_result.split("\n"):
                    if line.startswith("Full:"):
                        image_paths.append(line.replace("Full:", "").strip())
                print(f"[graph] Найдено изображений в TeX folder: {len(image_paths)}")

        for i, img_path in enumerate(image_paths, 1):
            print(f"[graph] --> Описание изображения {i}/{min(len(image_paths), 5)}")
            try:
                desc_state = {
                    "messages": [HumanMessage(content="Опиши это изображение для научной статьи")],
                    "base64_img": img_path,
                }
                result = self.describe_agent.run_with_state(desc_state)
                desc = result["messages"][-1].content
                image_descriptions.append(f"### Изображение {i} ({Path(img_path).name}):\n{desc}")
            except Exception as e:
                print(f"[graph] Ошибка описания изображения: {e}")
                image_descriptions.append(f"### Изображение {i}: Ошибка описания ({e})")

        print(f"[graph] <-- Описано изображений: {len(image_descriptions)}")

        return {
            **state,
            "all_image_descriptions": image_descriptions,
        }

    def _describe_images_for_write(self, state: MainState) -> MainState:
        """
        Найти и описать все изображения в статье перед написанием обзора.
        """
        tex_path = state.get("selected_paper_path")
        extracted_images_path = state.get("extracted_images_path")
        
        if not tex_path and not extracted_images_path:
            return {
                **state,
                "messages": [HumanMessage(content="Ошибка: путь к статье не найден.")],
                "all_image_descriptions": [],
            }

        image_descriptions = []
        image_paths = []

        # 1. Если есть изображения из PDF (extracted_images_path)
        if extracted_images_path:
            print(f"[graph] --> Использование изображений из PDF: {extracted_images_path}")
            img_folder = Path(extracted_images_path).parent
            if img_folder.exists():
                for img_ext in ["*.png", "*.jpg", "*.jpeg"]:
                    image_paths.extend(list(img_folder.glob(img_ext)))
                image_paths = [str(p) for p in image_paths]
                print(f"[graph] Найдено изображений в PDF folder: {len(image_paths)}")

        # 2. Если есть TeX-директория — ищем изображения там
        if not image_paths and tex_path:
            if "_tex" not in tex_path:
                tex_path = tex_path.replace(".pdf", "_tex")

            print(f"[graph] --> Поиск изображений для WRITE в {tex_path}")

            # Ищем изображения
            images_result = list_tex_images.invoke({"tex_path": tex_path})
            print(f"[graph] {images_result[:200]}")

            if "Изображения не найдены" not in images_result and "Ошибка" not in images_result:
                for line in images_result.split("\n"):
                    if line.startswith("Full:"):
                        image_paths.append(line.replace("Full:", "").strip())
                print(f"[graph] Найдено изображений в TeX folder: {len(image_paths)}")

        # Описываем каждое изображение (максимум 5)
        for i, img_path in enumerate(image_paths[:5], 1):
            print(f"[graph] --> Описание изображения {i}/{min(len(image_paths), 5)}")
            try:
                desc_state = {
                    "messages": [HumanMessage(content="Опиши это изображение для научной статьи")],
                    "base64_img": img_path,
                }
                result = self.describe_agent.run_with_state(desc_state)
                desc = result["messages"][-1].content
                image_descriptions.append(f"### Изображение {i} ({Path(img_path).name}):\n{desc}")
            except Exception as e:
                print(f"[graph] Ошибка описания изображения: {e}")
                image_descriptions.append(f"### Изображение {i}: Ошибка описания ({e})")

        print(f"[graph] <-- Описано изображений: {len(image_descriptions)}")

        return {
            **state,
            "all_image_descriptions": image_descriptions,
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
        image_descriptions = state.get("all_image_descriptions", [])
        print(f"[graph] --> EvalAgent (текст: {len(paper_to_eval)} символов, изображений: {len(image_descriptions)})")

        # Формируем объединённый контекст: сначала описания изображений, потом текст статьи
        if image_descriptions:
            combined_content = (
                f"=== ОПИСАНИЯ ИЗОБРАЖЕНИЙ ({len(image_descriptions)} шт.) ===\n" + "\n\n".join(image_descriptions) +
                f"\n\n=== ТЕКСТ СТАТЬИ ===\n{paper_to_eval}"
            )
        else:
            combined_content = paper_to_eval + "\n\n[ПРИМЕЧАНИЕ: Изображения не найдены или не описаны]"

        review_state = {"messages": combined_content}
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

    def _run_write_agent(self, state: MainState) -> MainState:
        paper_text = state.get("paper_content") or state["messages"][-1].content
        image_descriptions = state.get("all_image_descriptions", [])
        print(f"[graph] --> WriterAgent (текст: {len(paper_text)} символов, изображений: {len(image_descriptions)})")

        # Формируем объединённый контекст: сначала описания изображений, потом текст статьи
        if image_descriptions:
            combined_content = (
                f"=== ОПИСАНИЯ ИЗОБРАЖЕНИЙ ({len(image_descriptions)} шт.) ===\n" + "\n\n".join(image_descriptions) +
                f"\n\n=== ТЕКСТ СТАТЬИ ===\n{paper_text}"
            )
        else:
            combined_content = paper_text + "\n\n[ПРИМЕЧАНИЕ: Изображения не найдены или не описаны]"

        write_state = {"messages": [HumanMessage(content=combined_content)]}
        result = self.writer_agent.run_with_state(write_state)

        final_message = result["messages"][-1]
        print(f"[graph] <-- WriterAgent завершил работу")

        return {
            **state,
            "messages": [final_message],
            "written_review": final_message.content,
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
