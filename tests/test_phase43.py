"""Tests for Phase 43 — on-demand graph visualisation UI."""

from __future__ import annotations

from devagent.server.graph_ui import _PLACEHOLDER, _TEMPLATE, build_html


class TestBuildHtml:
    def test_returns_string(self) -> None:
        assert isinstance(build_html(), str)

    def test_injects_host_and_port(self) -> None:
        html = build_html("0.0.0.0", 9000)
        assert "http://0.0.0.0:9000" in html

    def test_default_base_url(self) -> None:
        html = build_html()
        assert "http://127.0.0.1:7331" in html

    def test_no_placeholder_remains(self) -> None:
        html = build_html("localhost", 1234)
        assert _PLACEHOLDER not in html

    def test_contains_doctype(self) -> None:
        assert build_html().startswith("<!doctype html>")

    def test_contains_d3_cdn(self) -> None:
        assert "cdnjs.cloudflare.com" in build_html()
        assert "d3" in build_html()

    def test_contains_api_endpoints(self) -> None:
        html = build_html()
        assert "/api/v1/sessions" in html
        assert "/api/v1/metrics" in html

    def test_contains_websocket_or_live(self) -> None:
        html = build_html()
        assert "Live" in html or "ldot" in html

    def test_all_placeholders_replaced(self) -> None:
        # template has multiple occurrences — all must be gone
        count_in_template = _TEMPLATE.count(_PLACEHOLDER)
        assert count_in_template >= 2, "template should have at least 2 placeholders"
        html = build_html("127.0.0.1", 7331)
        assert html.count(_PLACEHOLDER) == 0

    def test_api_docs_link_present(self) -> None:
        html = build_html("127.0.0.1", 7331)
        assert "/api/docs" in html

    def test_contains_connection_error_handling(self) -> None:
        html = build_html()
        # Verifies HTTP error checking and error propagation
        assert "if (!r.ok)" in html
        assert "Metrics request failed" in html
        assert "Sessions request failed" in html
        assert "setEndpointStatus" in html
        assert "connState" in html

    def test_contains_disconnected_ui_elements_and_styles(self) -> None:
        html = build_html()
        # Verifies disconnected state styling and UI indicators
        assert ".live-chip.disconnected" in html
        assert 'id="conn-banner"' in html
        assert 'id="conn-retry-btn"' in html
        assert "Disconnected" in html

    def test_contains_retry_and_refresh_wiring(self) -> None:
        html = build_html()
        # Verifies retry button click and auto-refresh wire to refreshAll
        assert "retryBtn.addEventListener('click'" in html
        assert "refreshAll()" in html


class TestFastapiGraphEndpoint:
    def test_get_root_returns_html(self) -> None:
        from fastapi.testclient import TestClient

        from devagent.server.fastapi_app import app

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "<!doctype html>" in r.text.lower()

    def test_get_root_contains_d3(self) -> None:
        from fastapi.testclient import TestClient

        from devagent.server.fastapi_app import app

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/")
        assert "d3" in r.text

    def test_get_root_not_in_openapi_schema(self) -> None:
        from fastapi.testclient import TestClient

        from devagent.server.fastapi_app import app

        client = TestClient(app)
        schema = client.get("/openapi.json").json()
        assert "/" not in schema.get("paths", {})
