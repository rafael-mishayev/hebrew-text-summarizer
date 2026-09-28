"""FastAPI application for the Hebrew Text Summarizer.

Exposes two routes that both render the same template: a GET that shows the
empty form and a POST that shows the result. Rendering server-side keeps the
application dependency-free on the client and means the page degrades to plain
HTML if JavaScript is unavailable.

All summarization work is delegated to :mod:`summarizer`; this module is
responsible only for HTTP concerns and for turning exceptions into messages
that are safe to display.
"""

import logging

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from summarizer import (
    MAX_INPUT_CHARACTERS,
    SummarizationError,
    SummarizationRateLimitError,
    SummaryLength,
    summarize_text,
)

# Configure the root logger before anything can emit a record, so provider
# failures logged inside summarizer.py actually reach the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Hebrew Text Summarizer")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the empty form.

    The default length and the input cap are passed in explicitly so the
    template never hard-codes values that the backend owns.
    """
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "selected_length": SummaryLength.MEDIUM.value,
            "max_input_characters": MAX_INPUT_CHARACTERS,
        },
    )


@app.post("/summarize", response_class=HTMLResponse)
async def summarize(
    request: Request,
    text: str = Form(...),
    # Typing the field as the enum makes FastAPI reject unknown values with a
    # 422 before this function body runs, so no unvalidated string reaches the
    # prompt-building code.
    length: SummaryLength = Form(SummaryLength.MEDIUM),
):
    """Summarize the submitted text and re-render the page with the result.

    Failures are rendered as an error message on the same page rather than
    raised, so the user keeps their input and can retry immediately.
    """
    cleaned_text = text.strip()
    summary = None
    error = None

    try:
        summary = await summarize_text(cleaned_text, length)
    except ValueError as exc:
        # Raised only for empty or oversized input. These messages describe
        # what the user must change and carry no internal detail, so they are
        # the one case where the exception text is shown verbatim.
        error = str(exc)
    except SummarizationRateLimitError:
        # Transient by nature, so the message invites an immediate retry.
        error = "שירות הסיכום עמוס כרגע. נסה שוב בעוד רגע."
    except SummarizationError:
        # Deliberately generic. The specific cause was already logged by the
        # summarizer; exposing it here would leak provider and configuration
        # details to anyone using the page.
        error = "שירות הסיכום אינו זמין כרגע. נסה שוב מאוחר יותר."

    # The original text and the selected length are echoed back so a failed or
    # successful submission does not clear the form.
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "summary": summary,
            "error": error,
            "original_text": cleaned_text,
            "selected_length": length.value,
            "max_input_characters": MAX_INPUT_CHARACTERS,
        },
    )
