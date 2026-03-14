from __future__ import annotations

from html import escape

import gradio as gr

from src.pipeline import RecipeRAGPipeline


pipeline = RecipeRAGPipeline()


def _build_list_html(items: list[str], empty_message: str) -> str:
    if not items:
        return f"<p>{empty_message}</p>"
    rendered = "".join(f"<li>{escape(item)}</li>" for item in items)
    return f"<ul>{rendered}</ul>"


def _format_rating(rating: float | None, vote_count: int | None) -> str:
    if rating is None:
        return "Not available"
    if vote_count is None:
        return f"{rating:.1f}/5"
    return f"{rating:.1f}/5 ({vote_count} votes)"


def run_recipe(query: str) -> tuple[str, str, str, str]:
    text = (query or "").strip()
    if not text:
        message = "<p>Please enter what you want to eat.</p>"
        return "", message, message, message

    result = pipeline.run(text)

    title = f"""
    <div class=\"result-title\">
      <h2>{escape(result.predicted_recipe)}</h2>
      <p><strong>Rating:</strong> {escape(_format_rating(result.rating, result.vote_count))}</p>
    </div>
    """

    ingredients_html = _build_list_html(result.ingredients, "No ingredients found.")
    steps_html = _build_list_html(result.summary_steps, "No steps found.")

    history_text = result.description or "History/description is not available for this recipe."
    history_html = f"<p>{escape(history_text)}</p>"

    return title, ingredients_html, steps_html, history_html


with gr.Blocks(theme=gr.themes.Soft(), title="ChefFri") as demo:
    gr.HTML(
        """
        <style>
          :root {
            --chef-bg: radial-gradient(circle at top left, #fff2db 0%, #fde0c3 45%, #f8cfaa 100%);
            --chef-card-bg: #fff8ef;
            --chef-border: #f3d4a1;
            --chef-text: #2a1a12;
            --chef-muted: #4f2d1f;
          }

          @media (prefers-color-scheme: dark) {
            :root {
              --chef-bg: radial-gradient(circle at top left, #1d140f 0%, #130e0a 45%, #0d0907 100%);
              --chef-card-bg: #231812;
              --chef-border: #7b573f;
              --chef-text: #fff4e8;
              --chef-muted: #f0cfb2;
            }
          }

          body.dark,
          .dark {
            --chef-bg: radial-gradient(circle at top left, #1d140f 0%, #130e0a 45%, #0d0907 100%);
            --chef-card-bg: #231812;
            --chef-border: #7b573f;
            --chef-text: #fff4e8;
            --chef-muted: #f0cfb2;
          }

          body {
            background: var(--chef-bg);
            color: var(--chef-text);
          }

          .gradio-container,
          .gradio-container label,
          .gradio-container input,
          .gradio-container textarea {
            color: var(--chef-text) !important;
            opacity: 1 !important;
            text-shadow: none !important;
            filter: none !important;
          }

          .gradio-container .prose,
          .gradio-container .prose p,
          .gradio-container .prose li,
          .gradio-container .prose h1,
          .gradio-container .prose h2,
          .gradio-container .prose h3,
          .gradio-container .prose strong,
          .gradio-container .prose span,
          .gradio-container .prose div,
          .gradio-container .gr-html,
          .gradio-container .gr-html * {
            color: var(--chef-text) !important;
            opacity: 1 !important;
            text-shadow: none !important;
            filter: none !important;
          }

          .gradio-container input,
          .gradio-container textarea,
          .gradio-container button,
          .gradio-container [data-testid="textbox"] textarea,
          .gradio-container [data-testid="textbox"] input {
            background: var(--chef-card-bg) !important;
            border-color: var(--chef-border) !important;
            color: var(--chef-text) !important;
            opacity: 1 !important;
          }

          .hero {
            text-align: center;
            margin-bottom: 8px;
          }
          .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            color: var(--chef-text);
          }
          .hero p {
            margin-top: 8px;
            font-size: 1rem;
            color: var(--chef-muted);
          }
          .result-title {
            background: var(--chef-card-bg);
            border: 1px solid var(--chef-border);
            border-radius: 14px;
            padding: 12px 16px;
            margin-bottom: 8px;
            color: var(--chef-text);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
          }
          .card {
            background: var(--chef-card-bg);
            border: 1px solid var(--chef-border);
            border-radius: 14px;
            padding: 14px 16px;
            min-height: 220px;
            color: var(--chef-text);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
          }
          .card h3 {
            margin-top: 0;
          }
          .card p,
          .card li {
            color: var(--chef-text);
          }
          .card ul {
            margin: 0;
            padding-left: 18px;
          }
          .history-card {
            margin-top: 12px;
            background: var(--chef-card-bg);
            border: 1px solid var(--chef-border);
            border-radius: 14px;
            padding: 14px 16px;
            color: var(--chef-text);
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
          }
          .history-card p,
          .history-card li,
          .result-title p,
          .result-title h2 {
            color: var(--chef-text);
          }

          .dark .card,
          .dark .history-card,
          .dark .result-title,
          .dark .hero h1,
          .dark .hero p {
            color: var(--chef-text) !important;
            opacity: 1 !important;
          }
        </style>
        <div class=\"hero\">
          <h1>Welcome to ChefFri <span aria-label=\"chef\" role=\"img\">👨‍🍳</span></h1>
          <p>Find your next meal in seconds.</p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        query = gr.Textbox(
            label="What you want to eat today",
            placeholder="Example: LITI CHOKHA",
            scale=5,
        )
        enter_button = gr.Button("Enter", variant="primary", scale=1)

    result_title = gr.HTML()

    with gr.Row(equal_height=True):
        ingredients_card = gr.HTML("<div class='card'><h3>Ingredients</h3><p>Enter a query to load ingredients.</p></div>")
        steps_card = gr.HTML("<div class='card'><h3>Steps</h3><p>Enter a query to load summarized steps.</p></div>")

    history_card = gr.HTML("<div class='history-card'><h3>Food History</h3><p>Enter a query to load food history/description.</p></div>")

    def wrap_cards(text: str) -> tuple[str, str, str, str]:
        title, ingredients, steps, history = run_recipe(text)
        return (
            title,
            f"<div class='card'><h3>Ingredients</h3>{ingredients}</div>",
            f"<div class='card'><h3>Steps</h3>{steps}</div>",
            f"<div class='history-card'><h3>Food History</h3>{history}</div>",
        )

    enter_button.click(
        fn=wrap_cards,
        inputs=[query],
        outputs=[result_title, ingredients_card, steps_card, history_card],
    )
    query.submit(
        fn=wrap_cards,
        inputs=[query],
        outputs=[result_title, ingredients_card, steps_card, history_card],
    )


if __name__ == "__main__":
  demo.launch(share=True)
