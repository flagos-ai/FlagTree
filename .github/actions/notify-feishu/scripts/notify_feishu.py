# -*- coding: utf-8 -*-

# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
notify_feishu.py - Post a card message to a Feishu chat

Generic on purpose: it renders whatever title, lines and link it is given and
knows nothing about FlagGems or any other producer. Deciding *whether* a run
is worth a message, and what to say about it, belongs to whoever has the
domain knowledge; this only delivers.

Takes either a notification JSON file:

    {"title": "...", "color": "red", "lines": [["Backend", "nvidia"], ...],
     "link": {"text": "Open run", "url": "https://..."}}

or the same fields on the command line.

Exits 0 even when sending fails, unless --strict: this runs inside a PR gate,
and a lost chat message must not fail someone's PR.

Usage:
    FEISHU_APP_ID=... FEISHU_APP_SECRET=... \\
    python notify_feishu.py --chat-id oc_xxx --notification notify.json

    # find the chat_id of a chat the bot was added to
    python notify_feishu.py --list-chats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import feishu_client as fc

# Feishu card header themes. Anything else is passed through, so a theme added
# by Feishu later needs no change here.
COLORS = ("blue", "wathet", "turquoise", "green", "yellow", "orange", "red", "carmine", "violet", "purple", "indigo",
          "grey")

DEFAULT_COLOR = "blue"

# A chat message that scrolls is a message nobody reads. Producers are
# expected to summarise; this is the backstop.
MAX_LINES = 20
MAX_VALUE_CHARS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chat-id", default=os.environ.get("FEISHU_CHAT_ID", ""),
                        help="Target chat_id (oc_...). Defaults to $FEISHU_CHAT_ID")
    parser.add_argument("--notification", type=Path, help="Path to a notification JSON file")
    parser.add_argument("--title", default="", help="Card title")
    parser.add_argument("--color", default="", help=f"Header theme, one of: {', '.join(COLORS)}")
    parser.add_argument("--text", default="", help="Body text, used when no --notification is given")
    parser.add_argument("--link-url", default="", help="URL for the button at the card footer")
    parser.add_argument("--link-text", default="Open", help="Label for that button")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when sending fails")
    parser.add_argument("--dry-run", action="store_true", help="Print the card that would be sent and exit")
    parser.add_argument("--list-chats", action="store_true", help="List the chats this bot belongs to, then exit")
    return parser.parse_args()


def load_notification(path: Optional[Path]) -> Dict:
    """Read a notification file, tolerating an absent one.

    A producer that decided there was nothing worth saying simply writes no
    file, so this is a normal outcome rather than an error.
    """
    if path is None:
        return {}
    if not path.is_file():
        print(f"No notification at {path}, nothing to send.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def _field(label: str, value: str) -> Dict:
    """One short label/value pair, laid out two per row."""
    text = str(value)
    if len(text) > MAX_VALUE_CHARS:
        text = text[:MAX_VALUE_CHARS - 1] + "…"
    return {
        "is_short": True,
        "text": {"tag": "lark_md", "content": f"**{label}**\n{text}"},
    }


def build_card(spec: Dict) -> Dict:
    """Render a notification spec into a Feishu interactive card.

    Lines come as [label, value] pairs rather than preformatted text so the
    card can lay them out, and so a producer does not have to know about
    Feishu's markup.
    """
    title = str(spec.get("title") or "CI notification")
    color = str(spec.get("color") or DEFAULT_COLOR)

    elements: List[Dict] = []

    lines = spec.get("lines") or []
    fields = []
    for item in lines[:MAX_LINES]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            fields.append(_field(item[0], item[1]))
        elif isinstance(item, dict) and "label" in item:
            fields.append(_field(item["label"], item.get("value", "")))
        else:
            # A bare string becomes its own paragraph: useful for a sentence
            # that does not fit the label/value shape.
            if fields:
                elements.append({"tag": "div", "fields": fields})
                fields = []
            elements.append({"tag": "div", "text": {"tag": "lark_md", "content": str(item)}})
    if fields:
        elements.append({"tag": "div", "fields": fields})

    text = spec.get("text")
    if text:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": str(text)}})

    note = spec.get("note")
    if note:
        elements.append({"tag": "note", "elements": [{"tag": "lark_md", "content": str(note)}]})

    link = spec.get("link") or {}
    if link.get("url"):
        elements.append({
            "tag":
            "action",
            "actions": [{
                "tag": "button",
                "text": {"tag": "plain_text", "content": str(link.get("text") or "Open")},
                "url": str(link["url"]),
                "type": "primary",
            }],
        })

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": color,
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": elements,
    }


def spec_from_args(args: argparse.Namespace) -> Dict:
    """Build a spec from the command line, for callers with one line to say."""
    spec: Dict = {}
    if args.title:
        spec["title"] = args.title
    if args.color:
        spec["color"] = args.color
    if args.text:
        spec["text"] = args.text
    if args.link_url:
        spec["link"] = {"text": args.link_text, "url": args.link_url}
    return spec


def do_list_chats() -> int:
    """Print chat_id and name for every chat the bot belongs to."""
    token = fc.get_token()
    chats = fc.list_chats(token)
    if not chats:
        print("This bot belongs to no chats. Add it to the target chat first:")
        print("  chat settings -> Group Bots -> Add Bot -> pick this app")
        return 1
    print(f"{'chat_id':<30} name")
    for chat in chats:
        print(f"{chat.get('chat_id', ''):<30} {chat.get('name', '')}")
    return 0


def main() -> int:
    args = parse_args()

    if args.list_chats:
        return do_list_chats()

    try:
        spec = load_notification(args.notification)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[ERROR] {e}")
        return 1 if args.strict else 0

    # Command-line values fill in what the file did not carry, so a producer
    # can emit the body and let the workflow supply the link.
    for key, value in spec_from_args(args).items():
        spec.setdefault(key, value)

    if not spec:
        # Nothing to say. The common case when a run had no change worth
        # reporting, so it is not a failure.
        return 0

    card = build_card(spec)

    if args.dry_run:
        print(json.dumps(card, ensure_ascii=False, indent=2))
        return 0

    if not args.chat_id:
        print("[WARN] No chat id given, nothing was sent.")
        return 1 if args.strict else 0

    if not fc.credentials_present():
        print("[WARN] FEISHU_APP_ID / FEISHU_APP_SECRET not set, nothing was sent.")
        return 1 if args.strict else 0

    try:
        token = fc.get_token()
        message_id = fc.send_message(token, args.chat_id, card)
    except fc.FeishuError as e:
        # A bot that was never added to the chat fails here, and the API's own
        # message does not say so. Name the likely cause.
        print(f"[ERROR] Sending failed: {e}")
        print("[HINT] If this is a permission error, check that the bot was "
              "added to the chat: chat settings -> Group Bots -> Add Bot.")
        return 1 if args.strict else 0

    print(f"[INFO] Sent message {message_id} to {args.chat_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
