from pathlib import Path
from typing import Iterable

import requests


def get_prompts() -> list[str]:
    """Возвращает 10 запросов, используемых в этом запуске инференса."""
    return [
        "Explain what a neural network is in one short paragraph.",
        "Give three practical tips for preparing for a programming exam.",
        "Write a Python function that checks whether a number is prime.",
        "Summarize the concept of gradient descent in simple words.",
        "Provide a 4-item checklist for debugging an HTTP API.",
        "What is overfitting in machine learning and how can it be reduced?",
        "Suggest a daily 30-minute plan to improve algorithm skills.",
        "Compare supervised and unsupervised learning in 5 bullet points.",
        "Create a short SQL query that returns top 5 products by sales.",
        "Explain the difference between CPU and GPU for AI inference.",
    ]


def query_ollama(
    prompt: str,
    model: str = "qwen2.5:0.5b",
    host: str = "http://127.0.0.1:11434",
    timeout_seconds: int = 180,
) -> str:
    """Отправляет один запрос в Ollama по HTTP и возвращает текст ответа модели."""
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()

    data = response.json()
    text = data.get("response")
    if not isinstance(text, str):
        raise ValueError("Ollama response JSON does not contain a string 'response' field.")

    return text.strip()


def run_inference(prompts: Iterable[str], model: str = "qwen2.5:0.5b") -> list[tuple[str, str]]:
    """Выполняет инференс для каждого запроса и возвращает пары запрос/ответ."""
    results: list[tuple[str, str]] = []

    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/10] Sending prompt...")
        answer = query_ollama(prompt=prompt, model=model)
        results.append((prompt, answer))

    return results


def escape_markdown_cell(text: str) -> str:
    """Экранирует спецсимволы таблицы Markdown и нормализует переносы строк."""
    escaped = text.replace("|", "\\|")
    return escaped.replace("\n", "<br>")


def write_markdown_report(rows: Iterable[tuple[str, str]], output_path: Path) -> None:
    """Записывает markdown-отчет инференса из двух столбцов: запрос и ответ модели."""
    lines = [
        "# Inference Report: qwen2.5:0.5b via Ollama",
        "",
        "| Prompt | LLM Output |",
        "|---|---|",
    ]

    for prompt, answer in rows:
        lines.append(f"| {escape_markdown_cell(prompt)} | {escape_markdown_cell(answer)} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Запускает полный процесс инференса и сохраняет markdown-отчет."""
    prompts = get_prompts()
    results = run_inference(prompts)

    report_path = Path("inference_report.md")
    write_markdown_report(results, report_path)
    print(f"Saved report to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
