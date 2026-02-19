import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from datetime import date
from agent_tools.tools import search_arxiv_papers, download_arxiv_paper, download_arxiv_tex, manage_files, parse_pdf_file

load_dotenv()

MAX_TOOL_CALLS = 8

class PaperState(TypedDict):
    messages: Annotated[list, add_messages]

class CoordinatorAgent:
    def __init__(self):
        self.tools = [search_arxiv_papers, download_arxiv_paper, download_arxiv_tex, parse_pdf_file]
        self.tools_no_download = [search_arxiv_papers]
        self.download_dir = "./downloads"

        _base_model = ChatOpenAI(
            model="arcee-ai/trinity-large-preview:free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            max_retries=2,
            max_tokens=4096,
            temperature=0.2,
        )
        self.model = _base_model.bind_tools(self.tools)
        self.model_no_download = _base_model.bind_tools(self.tools_no_download)
        
        self.date = date.today().isoformat()
        
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()
        
    def _build_graph(self):
        graph = StateGraph(PaperState)
        
        graph.add_node("process_query", self._process_query)
        graph.add_node("tools", self.tool_node)
        
        graph.add_edge(START, "process_query")
        graph.add_conditional_edges(
            "process_query",
            self._master_router,
            {
                "tools": "tools",
                "end": END
            }
        )
        graph.add_edge("tools", "process_query")
        
        return graph.compile()

    def _count_tool_calls(self, messages: list) -> int:
        return sum(1 for m in messages if isinstance(m, ToolMessage))


    def _process_query(self, state: PaperState) -> PaperState:
        tool_calls_used = self._count_tool_calls(state['messages'])
        remaining = MAX_TOOL_CALLS - tool_calls_used

        capabilities_block = """### ВОЗМОЖНОСТИ
- Поиск и скачивание: используй search_arxiv_papers, download_arxiv_paper, download_arxiv_tex.
- Управление файлами: используй manage_files для просмотра, чтения и поиска файлов (PDF, TeX, изображения).
- Специализированный анализ: делегируй задачи экспертным подагентам через ТЕГИ.
- Парсинг PDF: используй parse_pdf_file для извлечения текста из скачанного PDF."""
        delegation_note = "Для получения текста статьи перед [EVAL] или [WRITE]: используй ТОЛЬКО parse_pdf_file. После получения текста — ПРЕКРАТИ вызывать инструменты и сразу ответь."

        system_prompt = SystemMessage(
        content=f"""Ты — Science Helpy 3.0, старший научный ассистент и координатор мультиагентной системы для Telegram-канала «who is AI?».
Сегодня: {self.date}.

{capabilities_block}

### ГЛАВНАЯ ЦЕЛЬ
Твоя задача — удовлетворить запрос пользователя, используя инструменты и подагентов. Ты — "мозг" системы. Не зацикливайся на однотипных действиях. Если что-то не работает — меняй подход.

### АЛГОРИТМ РАБОТЫ (СТРОГО)

1. **АНАЛИЗ ЗАПРОСА:**
   - Пойми, что нужно пользователю: найти, скачать, оценить, написать обзор или всё сразу.
   - Если пользователь просит "последнюю" статью — используй `sort_strategy='submittedDate'`.
   - Если просит конкретную статью (напр. "DeepSeek-V3") — начни с `search_in_title_only=True`.

2. **ВЫПОЛНЕНИЕ ДЕЙСТВИЙ:**
   - **ШАГ 1: Поиск.** Сначала найди статью (`search_arxiv_papers`). Если не нашел — попробуй расширить запрос.
   - **ШАГ 2: Скачивание.** Если статья найдена и нужна для работы (оценка, обзор) — скачай её (`download_arxiv_paper`).
   - **ШАГ 3: Получение текста.** Чтобы подагенты могли работать, извлеки текст (`parse_pdf_file`).
   - **ШАГ 4: Делегирование.** Как только текст получен, СРАЗУ передавай задачу профильному подагенту (см. ТЕГИ).

3. **ПРАВИЛА ДЕЛЕГИРОВАНИЯ (ТЕГИ):**
   Теги — это команды передачи управления. Используй их в конце ответа, когда подготовил данные (текст или путь к картинке).

   - **[EVAL]** : Для оценки качества статьи.
     *Условие:* Текст статьи уже получен (`parse_pdf_file`).
     *Действие:* Напиши: "Передаю статью агенту-рецензенту для оценки качества. [EVAL]"

    - **[WRITE]**: Для написания подробного обзора.
      *Условие:* Пользователь ЯВНО попросил написать обзор ПОСЛЕ того, как ознакомился с оценкой.
      *Действие:* Напиши: "Передаю статью писателю для составления обзора. [WRITE]"

   - **[DESCRIBE]**: Для описания изображения.
     *Условие:* Есть путь к файлу изображения (найден через `manage_files`).
     *Действие:* Напиши: "Передаю изображение агенту компьютерного зрения. [DESCRIBE]"

   - **[END]**: Задача полностью выполнена.
     *Условие:* Ты получил ответ от подагента (оценку, обзор, описание) и показал его пользователю.
     *Действие:* Скопируй результат работы подагента в свой ответ и заверши тегом [END].

   **ВАЖНОЕ ПРАВИЛО ПОСЛЕДОВАТЕЛЬНОСТИ:**
   Сначала всегда делай [EVAL]. После оценки [EVAL] — ОСТАНОВИСЬ и покажи результат пользователю ([END]). НЕ переходи к [WRITE] автоматически, пока пользователь сам не попросит об этом в следующем сообщении.


### ЗАПРЕТЫ (ЧТОБЫ ИЗБЕЖАТЬ ЗАЦИКЛИВАНИЯ):
- **НЕ** скачивай одну и ту же статью дважды. Если файл уже есть (см. историю) — используй его.
- **НЕ** парси один и тот же PDF дважды. Внимательно смотри на сообщения из истории: если `parse_pdf_file` уже вызывался для этого файла, значит текст уже есть у тебя в памяти. Просто используй его (переходи к тегам).
- **НЕ** вызывай инструменты после того, как уже получил необходимые данные для делегирования. Сразу ставь тег.
- Если подагент вернул результат — **НЕ** отправляй его обратно тому же подагенту. Покажи результат пользователю и заверши работу ([END]) или предложи следующее действие.

### БЮДЖЕТ ИНСТРУМЕНТОВ:
Осталось {remaining} вызовов. Если лимит исчерпан — немедленно завершай работу тегом [END]."""
        )

        messages_to_send = [system_prompt] + state['messages']
        response = self.model.invoke(messages_to_send)

        return {
            "messages": [response]
        }
        
    def _master_router(self, state: PaperState) -> str:
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_calls_used = self._count_tool_calls(state['messages'])
            if tool_calls_used >= MAX_TOOL_CALLS:
                print(f"[coordinator] Tool call limit ({MAX_TOOL_CALLS}) reached, forcing end")
                return "end"
            for tc in last_message.tool_calls:
                args_preview = {k: (str(v)[:80] + "..." if len(str(v)) > 80 else v) for k, v in tc.get("args", {}).items()}
                print(f"[coordinator] tool call ({tool_calls_used + 1}/{MAX_TOOL_CALLS}): {tc['name']}({args_preview})")
            return "tools"
        
        return "end"
    
    def run(self, user_query: str):
        
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "download_dir": self.download_dir,
        }
        
        result = self.graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": "arxiv-agent-1"}}
        )

        return result
    
    def run_with_state(self, coord_state: PaperState):
        
        result = self.graph.invoke(
            coord_state,
            config={"recursion_limit": MAX_TOOL_CALLS * 2 + 4}
        )
        
        return result
