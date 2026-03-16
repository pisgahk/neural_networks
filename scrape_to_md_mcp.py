"""
scrape_to_md_mcp — MCP server for converting scraped JSON to Markdown
with free stock image replacement via Unsplash / Pexels.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Server init
# ---------------------------------------------------------------------------
mcp = FastMCP("scrape_to_md_mcp")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNSPLASH_API = "https://api.unsplash.com"
PEXELS_API = "https://api.pexels.com/v1"
DEFAULT_JSON_PATH = os.getenv(
       "DEFAULT_JSON_PATH",
       "/home/pisgah/dev/tests/rajesh/neural_networks/blogs.json"
   )

# Read API keys from environment — fall back to None (free/no-auth path)
UNSPLASH_ACCESS_KEY: Optional[str] = os.getenv("UNSPLASH_ACCESS_KEY")
PEXELS_API_KEY: Optional[str] = os.getenv("PEXELS_API_KEY")

# Fields that commonly hold an image URL inside scraped JSON
IMAGE_FIELD_HINTS = {
    "image", "img", "photo", "picture", "thumbnail", "banner",
    "cover", "hero", "avatar", "src", "url", "image_url", "photo_url",
    "picture_url", "img_src", "imgurl", "imageurl",
}

# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

class ConvertJsonInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    json_path: str = Field(
        ...,
        description=(
            "Absolute or relative path to the scraped .json file on disk. "
            "Example: '/home/user/scraped.json'"
        ),
    )
    output_path: Optional[str] = Field(
        default=None,
        description=(
            "Where to write the resulting .md file. "
            "Defaults to the same directory as json_path with a .md extension."
        ),
    )
    replace_images: bool = Field(
        default=True,
        description=(
            "When True, image URLs found in the JSON are replaced with a "
            "free stock photo that best matches the surrounding context."
        ),
    )
    image_source: str = Field(
        default="unsplash",
        description="Stock image provider: 'unsplash' or 'pexels'.",
        pattern="^(unsplash|pexels)$",
    )


class SearchImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Natural language search term for the stock image.",
        min_length=1,
        max_length=200,
    )
    source: str = Field(
        default="unsplash",
        description="Stock image provider: 'unsplash' or 'pexels'.",
        pattern="^(unsplash|pexels)$",
    )
    count: int = Field(
        default=1,
        description="Number of image results to return (1–5).",
        ge=1,
        le=5,
    )


# ---------------------------------------------------------------------------
# Helpers — image search
# ---------------------------------------------------------------------------

async def _search_unsplash(query: str, count: int) -> list[dict]:
    """Return up to `count` Unsplash image dicts: {url, alt, photographer}."""
    if not UNSPLASH_ACCESS_KEY:
        return _unsplash_demo_fallback(query)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{UNSPLASH_API}/search/photos",
            params={"query": query, "per_page": count, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for photo in data.get("results", [])[:count]:
        results.append(
            {
                "url": photo["urls"]["regular"],
                "alt": photo.get("alt_description") or query,
                "photographer": photo["user"]["name"],
                "source": "Unsplash",
                "source_url": photo["links"]["html"],
            }
        )
    return results


def _unsplash_demo_fallback(query: str) -> list[dict]:
    """Return a working Unsplash Source URL that needs no API key."""
    slug = query.strip().replace(" ", ",")
    url = f"https://source.unsplash.com/featured/1600x900/?{slug}"
    return [
        {
            "url": url,
            "alt": query,
            "photographer": "Unsplash",
            "source": "Unsplash (no-key)",
            "source_url": "https://unsplash.com",
        }
    ]


async def _search_pexels(query: str, count: int) -> list[dict]:
    """Return up to `count` Pexels image dicts."""
    if not PEXELS_API_KEY:
        return _pexels_demo_fallback(query)

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{PEXELS_API}/search",
            params={"query": query, "per_page": count, "orientation": "landscape"},
            headers={"Authorization": PEXELS_API_KEY},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for photo in data.get("photos", [])[:count]:
        results.append(
            {
                "url": photo["src"]["large"],
                "alt": photo.get("alt") or query,
                "photographer": photo["photographer"],
                "source": "Pexels",
                "source_url": photo["url"],
            }
        )
    return results


def _pexels_demo_fallback(query: str) -> list[dict]:
    """Pexels fallback when no API key is set — returns a placeholder."""
    slug = query.strip().replace(" ", "+")
    return [
        {
            "url": f"https://images.pexels.com/search/{slug}",
            "alt": query,
            "photographer": "Pexels",
            "source": "Pexels (no-key)",
            "source_url": "https://pexels.com",
        }
    ]


async def search_stock_image(
    query: str, source: str = "unsplash", count: int = 1
) -> list[dict]:
    """Unified image search dispatcher."""
    if source == "pexels":
        return await _search_pexels(query, count)
    return await _search_unsplash(query, count)


# ---------------------------------------------------------------------------
# Helpers — JSON → Markdown conversion
# ---------------------------------------------------------------------------

def _is_image_url(value: str) -> bool:
    """Heuristic: does this string look like an image URL?"""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        return False
    low = value.lower()
    return any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".avif")) or \
        any(hint in low for hint in ("image", "photo", "thumb", "banner", "cover", "cdn"))


def _is_image_field(key: str) -> bool:
    """Does the field name suggest it holds an image?"""
    return key.lower().strip("_- ") in IMAGE_FIELD_HINTS


def _extract_context_for_image(data: dict, current_key: str) -> str:
    """
    Build a meaningful search query from sibling fields like title, name,
    description, category — so the replacement image fits the content.
    """
    candidates = []
    for key in ("title", "name", "heading", "category", "topic", "tag", "label", "description"):
        val = data.get(key) or data.get(key.capitalize()) or data.get(key.upper())
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())
            if len(candidates) >= 2:
                break
    if not candidates:
        # Fall back to the key name itself
        candidates.append(current_key.replace("_", " "))
    return " ".join(candidates)[:150]


def _render_value(val: Any, depth: int = 0) -> str:
    """Recursively render a JSON value as Markdown text."""
    indent = "  " * depth
    if val is None:
        return "_empty_"
    if isinstance(val, bool):
        return "✅ Yes" if val else "❌ No"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        if not val:
            return "_none_"
        parts = []
        for item in val:
            parts.append(f"{indent}- {_render_value(item, depth + 1)}")
        return "\n".join(parts)
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            label = k.replace("_", " ").title()
            rendered = _render_value(v, depth + 1)
            if "\n" in rendered:
                parts.append(f"{indent}**{label}:**\n{rendered}")
            else:
                parts.append(f"{indent}**{label}:** {rendered}")
        return "\n".join(parts)
    return str(val)


def _to_title(key: str) -> str:
    return key.replace("_", " ").replace("-", " ").title()


def _find_page_title(data: dict) -> str:
    for k in ("title", "name", "heading", "h1", "page_title"):
        v = data.get(k) or data.get(k.capitalize())
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "Scraped Content"


async def _json_to_markdown(
    data: Any,
    replace_images: bool,
    image_source: str,
    _parent: Optional[dict] = None,
    _key: str = "",
) -> str:
    """
    Convert a (possibly nested) JSON structure to Markdown.
    Replaces image fields with stock photos when replace_images=True.
    """
    if isinstance(data, list):
        sections: list[str] = []
        for i, item in enumerate(data):
            chunk = await _json_to_markdown(item, replace_images, image_source, _parent=None, _key=str(i))
            sections.append(chunk)
        return "\n\n---\n\n".join(sections)

    if isinstance(data, dict):
        # Determine page title from top-level dict
        title = _find_page_title(data) if _parent is None else None
        lines: list[str] = []

        if title:
            lines.append(f"# {title}\n")

        for key, value in data.items():
            label = _to_title(key)

            # ---- image field handling ----
            if replace_images and (_is_image_field(key) or (isinstance(value, str) and _is_image_url(value))):
                context = _extract_context_for_image(data, key)
                images = await search_stock_image(context, image_source, 1)
                if images:
                    img = images[0]
                    lines.append(
                        f"## {label}\n\n"
                        f"![{img['alt']}]({img['url']})\n"
                        f"*Photo by [{img['photographer']}]({img['source_url']}) on {img['source']}*\n"
                    )
                else:
                    lines.append(f"## {label}\n\n_{value}_\n")
                continue

            # ---- recursive nested structures ----
            if isinstance(value, dict):
                nested = await _json_to_markdown(value, replace_images, image_source, _parent=data, _key=key)
                lines.append(f"## {label}\n\n{nested}\n")
                continue

            if isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(f"## {label}\n")
                for i, item in enumerate(value):
                    chunk = await _json_to_markdown(item, replace_images, image_source, _parent=data, _key=key)
                    lines.append(f"### Item {i + 1}\n\n{chunk}\n")
                continue

            # ---- plain fields ----
            rendered = _render_value(value)
            if key.lower() in ("title", "name", "heading", "h1") and _parent is None:
                continue  # already used as H1
            if "\n" in rendered:
                lines.append(f"## {label}\n\n{rendered}\n")
            else:
                lines.append(f"**{label}:** {rendered}\n")

        return "\n".join(lines)

    # Scalar root
    return str(data)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="convert_json_to_markdown",
    annotations={
        "title": "Convert Scraped JSON to Markdown",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def convert_json_to_markdown(params: ConvertJsonInput) -> str:
    """
    Read a scraped JSON file from disk, restructure it into clean, human-readable
    Markdown, and optionally replace all image URLs with matching free stock
    photos from Unsplash or Pexels.

    The output .md file is written to disk and its path is returned.

    Args:
        params (ConvertJsonInput): Validated input containing:
            - json_path (str): Path to the source .json file
            - output_path (Optional[str]): Destination .md file path
            - replace_images (bool): Swap image URLs for stock photos
            - image_source (str): 'unsplash' or 'pexels'

    Returns:
        str: JSON with keys 'output_path', 'preview' (first 500 chars), 'status'.
    """
    src = Path(params.json_path).expanduser().resolve()
    if not src.exists():
        return json.dumps({"error": f"File not found: {src}"})
    if not src.suffix.lower() == ".json":
        return json.dumps({"error": "json_path must point to a .json file."})

    try:
        raw = src.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}"})

    try:
        md = await _json_to_markdown(data, params.replace_images, params.image_source)
    except Exception as exc:
        return json.dumps({"error": f"Conversion failed: {exc}"})

    dest = Path(params.output_path).expanduser().resolve() if params.output_path else src.with_suffix(".md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")

    return json.dumps(
        {
            "status": "success",
            "output_path": str(dest),
            "preview": md[:500] + ("…" if len(md) > 500 else ""),
            "total_chars": len(md),
        },
        indent=2,
    )


@mcp.tool(
    name="search_stock_image",
    annotations={
        "title": "Search Free Stock Images",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def search_stock_image_tool(params: SearchImageInput) -> str:
    """
    Search Unsplash or Pexels for free stock images matching a query.
    Returns image URLs, attribution info, and source links.

    Args:
        params (SearchImageInput): Validated input containing:
            - query (str): Search keywords
            - source (str): 'unsplash' or 'pexels'
            - count (int): Number of results (1–5)

    Returns:
        str: JSON list of image objects with url, alt, photographer, source_url.
    """
    try:
        results = await search_stock_image(params.query, params.source, params.count)
        return json.dumps(results, indent=2)
    except httpx.HTTPStatusError as exc:
        return json.dumps({"error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"})
    except Exception as exc:
        return json.dumps({"error": f"Image search failed: {exc}"})


@mcp.tool(
    name="preview_json_structure",
    annotations={
        "title": "Preview JSON Structure",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def preview_json_structure(json_path: str) -> str:
    """
    Read a scraped JSON file and return a compact structural summary —
    top-level keys, value types, and detected image fields — without
    converting or writing anything.

    Args:
        json_path (str): Path to the .json file.

    Returns:
        str: JSON summary with 'keys', 'image_fields', 'item_count' (if list).
    """
    src = Path(json_path).expanduser().resolve()
    if not src.exists():
        return json.dumps({"error": f"File not found: {src}"})

    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}"})

    def _summarise(obj: Any, depth: int = 0) -> dict:
        if isinstance(obj, dict):
            keys = {}
            for k, v in obj.items():
                type_name = type(v).__name__
                is_img = _is_image_field(k) or (isinstance(v, str) and _is_image_url(v))
                keys[k] = {"type": type_name, "is_image": is_img}
                if isinstance(v, dict) and depth < 2:
                    keys[k]["children"] = _summarise(v, depth + 1)
            return keys
        if isinstance(obj, list):
            return {"list_length": len(obj), "first_item": _summarise(obj[0], depth) if obj else None}
        return {"value": str(obj)[:80]}

    summary = _summarise(data)
    img_fields = [k for k, v in (summary.items() if isinstance(summary, dict) else []) if v.get("is_image")]

    return json.dumps(
        {"structure": summary, "detected_image_fields": img_fields, "root_type": type(data).__name__},
        indent=2,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
