"""07 · Images: typed questions about a screenshot, then a browser agent where Jev picks each click.

    uv run python 07_images_and_browser_agent.py screenshot
        # a Noul, a Choice, a Score and a Box about a LangChain docs screenshot
    uv run --extra browser-agent python 07_images_and_browser_agent.py agent
        # Browser Use's jev-ultrafast on the LangChain docs, then a Jev check of the final screen

Needs TYPESAFE_API_KEY and OPENAI_API_KEY in .env. For a LangSmith trace of each Jev request,
also set LANGSMITH_TRACING=true and LANGSMITH_API_KEY. The agent also needs Python 3.12+, Chrome
with remote debugging allowed, and the browser-agent extra (jev-ultrafast).
Images and Box are "Dreamer-ready" in the TypeSafe docs: see https://docs.typesafe.ai/images.
"""

import base64
import io
import json
import os
import sys
import time
import uuid

import httpx
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.schemas import Attachment
from PIL import Image

load_dotenv()

SCREENSHOT = "auxiliary/langchain_docs.png"  # docs.langchain.com, 1440 x 900, captured 2026-11-04


@traceable(name="jev_systemone")  # traced to LangSmith when LANGSMITH_TRACING=true
def ask_jev(screen: Attachment, questions: dict) -> dict:
    """One System One request with one image in the state."""
    image = Image.open(io.BytesIO(screen.data)).convert("RGB")
    image.thumbnail((1568, 1568))  # each side must be 280-1568 px; the API never resizes
    buf = io.BytesIO()
    image.save(buf, "PNG")

    ref = f"{uuid.uuid4()}-screen"  # a fresh reference for each image in each request
    body = {
        "model": "jev-latest",
        "state": {"screen": ref},
        "assets": {ref: {"type": "image", "format": "png",
                         "data_base64": base64.b64encode(buf.getvalue()).decode()}},
        "questions": questions,
    }
    response = httpx.post(
        "https://api.typesafe.ai/v1/systemone",
        json=body,
        headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                 "TypeSafe-Danger-Zone-Images": "true"},  # without it, 400
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    print(f"model {result['model']}  request {response.headers.get('x-typesafe-request-id')}"
          f"  input tokens {result['usage']['input_tokens']}  image sent {image.width}x{image.height}")
    return result["answers"]


QUESTIONS = {
    "search_box_visible": {
        "type": "noul",
        "instructions": "Is there a search box in `screen`?",
    },
    "page_kind": {
        "type": "choice",
        "instructions": "What kind of page is `screen`?",
        "criteria": {
            "docs_home": "The landing page of a documentation site, linking to its main sections.",
            "docs_article": "One documentation article that explains a single topic.",
            "api_reference": "Reference tables of classes, functions, and parameters.",
            "blog": "A blog post or news article.",
            "login": "A sign-in form.",
        },
    },
    "beginner_friendly": {
        "type": "score",
        "instructions": "How easy is it for a newcomer to know where to start from `screen`?",
        "criteria": [
            "No clear starting point: dense reference material only.",
            "A starting point exists but is hard to find.",
            "Clear sections, each with a way in.",
            "An obvious first step, such as a prominent 'Get started' link.",
        ],
    },
    "search_box": {
        "type": "box",
        "instructions": "the search box in `screen`",
    },
}


def screenshot() -> None:
    with open(SCREENSHOT, "rb") as f:
        data = f.read()
    a = ask_jev(Attachment("image/png", data), QUESTIONS)
    print(json.dumps(a, indent=2))

    w, h = Image.open(io.BytesIO(data)).size
    if a["search_box_visible"]["noul"] < 0.5:
        print("no search box, so the Box answer points at something else: ignored")
        return
    b = a["search_box"]["box"]
    pixels = [round(b["left"] * w), round(b["top"] * h), round(b["right"] * w), round(b["bottom"] * h)]
    print(f"search box at {pixels} px (left, top, right, bottom)")


# The text helper in jev-ultrafast speaks OpenRouter's dialect, which OpenAI's API rejects.
# The library is open source, so we swap in a small OpenAI version of the same function.
def openai_field_text(context: dict) -> tuple[str, dict]:
    from openai import OpenAI
    from jev_ultrafast.questions import TEXT_VALUE

    started = time.perf_counter()
    reply = OpenAI().chat.completions.create(
        model="gpt-6-luna",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": TEXT_VALUE},
                  {"role": "user", "content": json.dumps(context)}],
    )
    text = json.loads(reply.choices[0].message.content)["text"]
    return text, {"model": reply.model, "latency_ms": round((time.perf_counter() - started) * 1000),
                  "usage": reply.usage.model_dump()}


def agent() -> None:
    import jev_ultrafast.agent
    from jev_ultrafast import Agent

    os.environ.setdefault("TEXT_MODEL_API_KEY", os.environ["OPENAI_API_KEY"])
    jev_ultrafast.agent.field_text = openai_field_text

    goal = ("Search the docs for structured output and open the page about it. "
            "Stop when that page is visible.")
    with Agent("https://docs.langchain.com", goal, record_dir="auxiliary/out/agent_frames") as run:
        done = 0
        for state in run.run():
            for step in state["history"][done:]:  # print each action once
                print(f"{step['executed_ms']:>6} ms  {step['operation']:<9} {step['action'][:40]!r}"
                      f"  p={step['probability']:.2f}  {step['text'] or ''}")
            done = len(state["history"])
        final = run.state
        print(f"\n{final['status']} in {final['elapsed_ms']} ms, {len(final['decisions'])} Jev requests, "
              f"{len(final['text_calls'])} text calls, at {final['page']['url']}")
        jev_ms = [d["latency_ms"] for d in final["decisions"]]
        text_ms = [t["latency_ms"] for t in final["text_calls"]]
        print(f"Jev latency per request {jev_ms} ms, text calls {text_ms} ms")
        last_screen = base64.b64decode(final["page"]["screenshot"])

    # The agent's DONE is its own claim. Check the final screen with a question of our own.
    check = ask_jev(Attachment("image/jpeg", last_screen), {"on_target": {
        "type": "noul",
        "instructions": "Is `screen` a documentation page about structured output?",
    }})
    print(f"on_target {check['on_target']['noul']:.2f}")
    with open("auxiliary/out/agent_final.jpg", "wb") as f:
        f.write(last_screen)


if __name__ == "__main__":
    {"screenshot": screenshot, "agent": agent}[sys.argv[1]]()
