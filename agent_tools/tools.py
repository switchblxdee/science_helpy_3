import os
import re
import json
import tarfile
import requests
import io
from datetime import datetime
import arxiv
from langchain.tools import tool
import pymupdf4llm
from pathlib import Path

DOWNLOAD_DIR = Path("./downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

@tool("search_arxiv_papers")
def search_arxiv_papers(
    query: str, 
    limit: int = 20,
    sort_strategy: str = "relevance",
    date_from: str = None,
    date_to: str = None,
    search_in_title_only: bool = False
) -> str:
    """
    Search for academic papers on arXiv.
    
    Args:
        query: Topic (e.g. 'DeepSeek', 'LLM', "Sparse Attention"). 
               You can use queries like "Qwen3" or "Qwen 3". "DeepSeek-V3" or "DeepSeek V3" etc.
        limit: Max results (default 20). Increase to 20-30 for broad topics.
        sort_strategy: 'relevance' (default), 'submittedDate' (newest), 'lastUpdatedDate'.
        date_from: Start date 'YYYY-MM-DD'.
        date_to: End date 'YYYY-MM-DD'.
        search_in_title_only: Set True to find official/main papers (e.g. "DeepSeek-V3"). 
                              Helps filter out papers that just *mention* the query. Try in after of first search_arxiv_papers.
    """
    try:
        client = arxiv.Client()
        
        sort_criterion = arxiv.SortCriterion.Relevance
        if sort_strategy == "submittedDate":
            sort_criterion = arxiv.SortCriterion.SubmittedDate
        elif sort_strategy == "lastUpdatedDate":
            sort_criterion = arxiv.SortCriterion.LastUpdatedDate

        processed_query = query
        if search_in_title_only:
            words = query.strip().split()
            if len(words) == 1:
                processed_query = f"ti:{words[0]}"
            else:
                processed_query = " AND ".join(f"ti:{w}" for w in words)
        
        if date_from:
            start_str = date_from.replace("-", "") + "0000"
            if date_to:
                end_str = date_to.replace("-", "") + "2359"
            else:
                end_str = datetime.now().strftime("%Y%m%d%2359")
            
            # Важно: берем query в скобки, чтобы дата применялась ко всему запросу
            processed_query = f'({processed_query}) AND submittedDate:[{start_str} TO {end_str}]'

        search = arxiv.Search(
            query=processed_query,
            max_results=limit,
            sort_by=sort_criterion
        )
        
        results = []
        for paper in client.results(search):
            raw_id = paper.get_short_id()
            results.append({
                "arxiv_id": raw_id,
                "title": paper.title,
                "published": paper.published.strftime("%Y-%m-%d"),
                "authors": [a.name for a in paper.authors[:3]],
                "summary": paper.summary[:400].replace("\n", " ") + "..."
            })

        if not results:
            return f"Статьи по запросу '{processed_query}' не найдены. Попробуй изменить запрос или отключить флаг search_in_title_only."

        lines = [f"Найдено статей: {len(results)} (Запрос: {processed_query})\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. ИСПОЛЬЗУЙ ЭТОТ ID ДЛЯ СКАЧИВАНИЯ: {r['arxiv_id']}\n"
                f"   Название: {r['title']}\n"
                f"   Авторы: {', '.join(r['authors'])}\n"
                f"   Дата: {r['published']}\n"
                f"   Аннотация: {r['summary']}\n"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"Error: {str(e)}"
    
@tool("download_arxiv_paper")
def download_arxiv_paper(arxiv_id: str) -> str:
    """
    Download a specific arXiv paper PDF by its ID.
    Args:
        arxiv_id: The short ID of the paper (e.g., '1706.03762').
                  ВАЖНО: используй ТОЧНЫЙ ID из результатов поиска search_arxiv_papers.
    """
    try:
        clean_id = re.sub(r'v\d+$', '', arxiv_id.strip())

        client = arxiv.Client()
        search = arxiv.Search(id_list=[clean_id])
        paper = next(client.results(search))

        file_name = f"{clean_id}.pdf"
        paper.download_pdf(dirpath=str(DOWNLOAD_DIR), filename=file_name)

        full_path = DOWNLOAD_DIR / file_name
        result = {
            "status": "success",
            "arxiv_id": clean_id,
            "path": str(full_path.absolute()),
            "title": paper.title,
            "authors": [a.name for a in paper.authors[:3]],
            "published": paper.published.strftime("%Y-%m-%d"),
        }
        print(f"[download_arxiv_paper] Скачано: '{paper.title}' (ID: {clean_id})")
        
        return json.dumps({
            "status": "success",
            "arxiv_id": clean_id,
            "path": str(full_path.absolute()),
            "title": paper.title,
            "authors": [a.name for a in paper.authors[:3]],
            "published": paper.published.strftime("%Y-%m-%d"),
            "message": f"Статья '{paper.title}' успешно скачана."
        }, ensure_ascii=False)

    except StopIteration:
        return f"Статья с ID '{arxiv_id}' не найдена на arXiv. Проверь ID — он должен быть из результатов поиска."
    except Exception as e:
        return f"Ошибка при скачивании '{arxiv_id}': {str(e)}"
    
@tool("download_arxiv_tex")
def download_arxiv_tex(arxiv_id: str) -> str:
    """
    Downloads the LaTeX source files for a specific arXiv paper.
    Args:
        arxiv_id: The short ID of the paper (e.g., '1706.03762').
    """
    try:
        paper_dir = DOWNLOAD_DIR / f"{arxiv_id}_tex"
        paper_dir.mkdir(exist_ok=True)

        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        
        response = requests.get(source_url, stream=True, timeout=20)
        response.raise_for_status()
        
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=paper_dir)

        tex_files = list(paper_dir.glob("**/*.tex"))
        
        return json.dumps({
            "status": "success",
            "directory": str(paper_dir.absolute()),
            "main_tex_files": [f.name for f in tex_files],
            "message": f"Source files extracted to {paper_dir}"
        }, ensure_ascii=False)

    except requests.exceptions.RequestException as e:
        return f"Network error while downloading {arxiv_id}: {str(e)}"
    except tarfile.ReadError:
        return f"Error: Paper {arxiv_id} does not have TeX sources (it might be a direct PDF upload)."
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# @tool("manage_files")
# def manage_files(action: str, content: str = None, destination: str = None, path: str = "/Users/switchblade/Documents/vs_code/science_helpy_3/downloads") -> str:
#     """
#     Универсальный инструмент для работы с файлами в рабочей директории.
    
#     Args:
#         action: 'list' (показать файлы), 'read' (читать), 'write' (создать/редактировать), 
#                 'move' (переименовать/переместить), 'delete' (удалить).
#         path: Путь к файлу или папке.
#         content: Текст для записи (только для action='write').
#         destination: Новый путь (только для action='move').
#     """
#     p = Path(path)
    
#     try:
#         if action == "list":
#             if not p.exists(): return f"Ошибка: Путь {path} не существует."
#             items = []
#             for item in p.iterdir():
#                 items.append(f"{item.name}")
#             return "\n".join(items) if items else "Папка пуста."

#         elif action == "read":
#             if not p.is_file(): return f"Ошибка: {path} не является файлом."
#             binary_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
#             if p.suffix.lower() in binary_extensions:
#                 return (
#                     f"Ошибка: файл '{p.name}' является бинарным ({p.suffix}) и не может быть прочитан как текст. "
#                     f"Для PDF используй инструмент parse_pdf_file. "
#                     f"Для изображений используй агента [DESCRIBE]."
#                 )
#             with open(p, 'r', encoding='utf-8', errors='ignore') as f:
#                 content = f.read()
#             MAX_READ_CHARS = 20_000
#             if len(content) > MAX_READ_CHARS:
#                 content = content[:MAX_READ_CHARS] + f"\n\n[...обрезано, показано {MAX_READ_CHARS} из {len(content)} символов]"
#             return content

#         elif action == "write":
#             p.parent.mkdir(parents=True, exist_ok=True)
#             with open(p, 'w', encoding='utf-8') as f:
#                 f.write(content or "")
#             return f"Файл {path} успешно записан."

#         elif action == "move":
#             if not destination: return "Ошибка: Укажите 'destination' для перемещения."
#             dest_path = Path(destination)
#             dest_path.parent.mkdir(parents=True, exist_ok=True)
#             p.rename(dest_path)
#             return f"Перемещено из {path} в {destination}."

#         elif action == "delete":
#             if p.is_dir():
#                 import shutil
#                 shutil.rmtree(p)
#             else:
#                 p.unlink()
#             return f"Объект {path} удален."

#         return "Ошибка: Неизвестное действие."

#     except Exception as e:
#         return f"Ошибка при работе с файлом: {str(e)}"
    
MAX_PDF_CHARS = 60_000

@tool("parse_pdf_file")
def parse_pdf_file(pdf_path: str) -> str:
    """
    Parses a PDF file and extracts its text content.
    Returns up to 60 000 characters to avoid context window overflow.
    
    Args:
        pdf_path: The path to the PDF file to be parsed.
    """
    
    markdown_text = pymupdf4llm.to_markdown(pdf_path)
    if len(markdown_text) > MAX_PDF_CHARS:
        markdown_text = markdown_text[:MAX_PDF_CHARS] + f"\n\n[...текст обрезан, показано {MAX_PDF_CHARS} из {len(markdown_text)} символов]"
    return markdown_text
    
