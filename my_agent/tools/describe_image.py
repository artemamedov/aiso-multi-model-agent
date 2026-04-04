"""Tool that uses a local vision model to describe image contents."""
import base64
import io
import os
import re

import requests
from PIL import Image


def describe_image(image_path: str, query: str) -> str:
    """Describe the contents of an image using a local vision model.

    Use this tool when you need to understand what's in an image.
    Ask a specific question about the image to get focused, useful information.

    Args:
        image_path: Path to the image file.
        query: What you want to know about the image. Be specific.
               Example: "List all plan names, prices, and storage limits shown in this pricing table."

    Returns:
        A text description of the relevant image contents.
    """
    api_base = os.getenv("LMSTUDIO_API_BASE", "http://127.0.0.1:1234/v1")
    vision_model = os.getenv("OLLAMA_VISION_MODEL", "qwen/qwen3-vl-4b")

    # Load and resize image
    img = Image.open(image_path)
    max_dim = 1024
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            headers={"Authorization": "Bearer lm-studio"},
            json={
                "model": vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 600,
                "temperature": 0,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return f"Error: Vision model returned HTTP {resp.status_code}"

        content = resp.json()["choices"][0]["message"]["content"]
        # Strip think blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        return f"Error calling vision model: {e}"
