"""HTTP-level tests for the routes in :mod:`main`."""

import re

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def checked_length(html):
    return re.search(r'value="(\w+)"\s*checked', html).group(1)


def test_home_renders_an_rtl_form_with_medium_selected(client):
    response = client.get("/")

    assert response.status_code == 200
    assert '<html dir="rtl" lang="he">' in response.text
    assert '<label for="text"' in response.text
    assert checked_length(response.text) == "medium"


def test_summary_is_rendered_and_the_form_keeps_its_input(client, fake_client):
    fake_client.content = "זהו הסיכום."

    response = client.post("/summarize", data={"text": "טקסט מקור", "length": "short"})

    assert response.status_code == 200
    assert "זהו הסיכום." in response.text
    assert ">טקסט מקור</textarea>" in response.text
    assert checked_length(response.text) == "short"


def test_model_output_is_html_escaped(client, fake_client):
    fake_client.content = "<script>alert(1)</script>"

    response = client.post("/summarize", data={"text": "טקסט"})

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert "<script>alert(1)</script>" not in response.text


def test_unknown_length_is_rejected_before_any_request(client, fake_client):
    response = client.post("/summarize", data={"text": "טקסט", "length": "huge"})

    assert response.status_code == 422
    assert fake_client.calls == []


def test_blank_text_shows_a_hebrew_validation_message(client, fake_client):
    response = client.post("/summarize", data={"text": "   ", "length": "detailed"})

    assert "הטקסט ריק" in response.text
    assert checked_length(response.text) == "detailed"
    assert fake_client.calls == []


def test_rate_limit_shows_a_retry_message(client, fake_client):
    fake_client.fail_with_rate_limit()

    response = client.post("/summarize", data={"text": "טקסט"})

    assert "שירות הסיכום עמוס כרגע" in response.text


def test_provider_failure_does_not_leak_internal_details(client, fake_client):
    fake_client.fail_with_connection_error("upstream detail: key sk-or-secret")

    response = client.post("/summarize", data={"text": "טקסט"})

    assert "שירות הסיכום אינו זמין כרגע" in response.text
    assert "sk-or-secret" not in response.text
    assert "upstream detail" not in response.text
