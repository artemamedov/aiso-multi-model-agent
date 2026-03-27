"""
Google ADK Agent Runner using API Server.
This module provides a function to run the ADK agent via HTTP requests to the API server.
"""

import atexit
import os
import subprocess
import time
import uuid
from typing import Any

import requests


class ADKAgentRunner:
    """
    Client for interacting with ADK agent via FastAPI server.
    Automatically starts and manages the API server lifecycle.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        agent_name: str = "my_agent",
        user_id: str = "dev_user",
    ):
        self.base_url = base_url
        self.agent_name = agent_name
        self.user_id = (
            user_id  # User ID for sessions (use same as web UI to see eval chats there)
        )
        self.server_process = None
        self._we_started_server = False  # Track if we started the server

    def _is_server_running(self) -> bool:
        """Check if an ADK API server is already running."""
        try:
            response = requests.get(f"{self.base_url}/list-apps", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start_server(self):
        """Start the ADK API server in the background if not already running."""
        # First check if a server is already running
        if self._is_server_running():
            print(f"✓ Using existing ADK API server at {self.base_url}")
            return

        if self.server_process is not None:
            print("Server process already started by us")
            return

        print("Starting ADK API server...")
        # Start the server in the background
        self.server_process = subprocess.Popen(
            ["adk", "api_server", "--host", "127.0.0.1", "--port", "8000", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )

        self._we_started_server = True

        # Register cleanup on exit (only if we started it)
        atexit.register(self.stop_server)

        # Wait for server to be ready
        max_retries = 30
        for _ in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/list-apps", timeout=1)
                if response.status_code == 200:
                    print(f"✓ ADK API server started successfully on {self.base_url}")
                    return
            except requests.exceptions.RequestException:
                time.sleep(1)

        raise RuntimeError("Failed to start ADK API server")

    def stop_server(self):
        """Stop the ADK API server (only if we started it)."""
        if self.server_process is not None and self._we_started_server:
            print("\nStopping ADK API server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            self._we_started_server = False

    def restart_server(self):
        """Gracefully stop the server and start a fresh one."""
        print("\nRestarting ADK API server after timeout...")
        if self.server_process is not None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait(timeout=3)
            self.server_process = None
        self._we_started_server = False
        time.sleep(2)
        self.start_server()

    @staticmethod
    def _extract_response_details(
        events: list[dict[str, Any]],
    ) -> tuple[str, list[str]]:
        """Extract response text and tool call names from ADK events.

        Some model/tool interactions may stop after a tool response without
        emitting text content. In that case, fall back to the latest tool result.
        """
        response_parts: list[str] = []
        tool_calls: list[str] = []
        tool_results: list[str] = []

        for event in events:
            content = event.get("content")
            if not content:
                continue
            parts = content.get("parts", [])
            for part in parts:
                text = part.get("text")
                if text:
                    response_parts.append(text)

                function_call = part.get("functionCall")
                if function_call:
                    name = function_call.get("name", "")
                    if name:
                        tool_calls.append(name)

                function_response = part.get("functionResponse")
                if function_response:
                    response_payload = function_response.get("response")
                    if isinstance(response_payload, dict) and "result" in response_payload:
                        result_text = str(response_payload.get("result", "")).strip()
                    else:
                        result_text = str(response_payload).strip()
                    if result_text:
                        tool_results.append(result_text)
        final_text = "".join(response_parts).strip()
        if not final_text and tool_results:
            final_text = tool_results[-1]
        return final_text, tool_calls

    def run_agent(
        self, question: str, file_paths: list[str] | None = None
    ) -> dict[str, Any]:
        """Run agent and return response text plus tool calls metadata."""
        if self.server_process is None and not self._is_server_running():
            self.start_server()

        session_id = f"eval_{uuid.uuid4().hex[:12]}"

        try:
            session_response = requests.post(
                f"{self.base_url}/apps/{self.agent_name}/users/{self.user_id}/sessions/{session_id}",
                json={"state": {}},
                timeout=10,
            )
            session_response.raise_for_status()
        except requests.exceptions.RequestException as error:
            if self._we_started_server:
                self.restart_server()
            raise RuntimeError(
                f"Failed to create session for agent '{self.agent_name}': {error}"
            ) from error

        message_parts = [{"text": question}]
        if file_paths:
            import base64
            from pathlib import Path
            for fp in file_paths:
                p = Path(fp)
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    with open(p, "rb") as img_file:
                        b64 = base64.b64encode(img_file.read()).decode()
                    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
                    message_parts.append({
                        "inline_data": {
                            "mime_type": mime.get(p.suffix.lower().lstrip("."), "image/png"),
                            "data": b64,
                        }
                    })
                else:
                    message_parts[0]["text"] += f"\n\nNote: The following files are relevant: {fp}"

        # Send message using /run endpoint
        try:
            response = requests.post(
                f"{self.base_url}/run",
                json={
                    "app_name": self.agent_name,
                    "user_id": self.user_id,
                    "session_id": session_id,
                    "new_message": {
                        "role": "user",
                        "parts": message_parts,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            events = response.json()
            response_text, tool_calls = self._extract_response_details(events)
            return {
                "response_text": response_text,
                "tool_calls": tool_calls,
                "session_id": session_id,
            }
        except requests.exceptions.RequestException as error:
            if self._we_started_server:
                self.restart_server()
            error_message = str(error)
            if hasattr(error, "response") and error.response is not None:
                body = error.response.text.strip()
                if body:
                    error_message = f"{error_message} | body: {body[:500]}"
            raise RuntimeError(
                f"Failed to run agent on question: {error_message}"
            ) from error


# Global runner instance
_runner = None


def run_agent(
    question: str, file_paths: list[str] | None = None, user_id: str = "dev_user"
) -> dict[str, Any]:
    """
    Run the Google ADK agent on a given question.
    This function manages the API server lifecycle automatically.

    Args:
        question: The question to answer
        file_paths: Optional list of file paths that may be needed to answer the question
        user_id: User ID for the session (default: "dev_user"). Use same as web UI to see eval chats there.

    Returns:
        The agent's response as a string
    """
    global _runner

    if _runner is None:
        _runner = ADKAgentRunner(user_id=user_id)
        _runner.start_server()

    return _runner.run_agent(question, file_paths)
