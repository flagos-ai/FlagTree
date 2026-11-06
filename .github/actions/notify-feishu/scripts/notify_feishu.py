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
     "links": [{"text": "Open run", "url": "https://..."}, ...]}

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
    parser.add_argument(
        "--open-id", default=os.environ.get("FEISHU_OPEN_ID", ""),
        help=("Target user open_id(s) (ou_...) for a direct message, comma "
              "separated for several. Takes precedence over --chat-id. "
              "Defaults to $FEISHU_OPEN_ID"))
    parser.add_argument("--notification", type=Path, help="Path to a notification JSON file")
    parser.add_argument("--title", default="", help="Card title")
    parser.add_argument("--color", default="", help=f"Header theme, one of: {', '.join(COLORS)}")
    parser.add_argument("--text", default="", help="Body text, used when no --notification is given")
    parser.add_argument("--link-url", default="", help="URL for the button at the card footer")
    parser.add_argument("--link-text", default="Open", help="Label for that button")
    parser.add_argument("--link2-url", default="", help="URL for a second button, when there is one")
    parser.add_argument("--link2-text", default="Open", help="Label for the second button")
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

    # `link` for one button, `links` for a row of them. The first is primary
    # so there is still an obvious default action among several.
    links = spec.get("links") or []
    if not links:
        single = spec.get("link") or {}
        if single:
            links = [single]
    buttons = []
    for index, item in enumerate(links):
        if not isinstance(item, dict) or not item.get("url"):
            continue
        buttons.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": str(item.get("text") or "Open")},
            "url": str(item["url"]),
            "type": "primary" if index == 0 else "default",
        })
    if buttons:
        elements.append({"tag": "action", "actions": buttons})

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
    links = []
    if args.link_url:
        links.append({"text": args.link_text, "url": args.link_url})
    if args.link2_url:
        links.append({"text": args.link2_text, "url": args.link2_url})
    if links:
        spec["links"] = links
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

    # A direct message wins over a chat: a caller that resolved a person meant
    # to reach that person, not to fall back to a room full of others. Several
    # ids are allowed, since one failure can concern more than one person.
    if args.open_id:
        receive_id_type = "open_id"
        recipients = [p for p in args.open_id.replace(",", " ").split() if p]
    elif args.chat_id:
        receive_id_type = "chat_id"
        recipients = [args.chat_id]
    else:
        print("[WARN] No recipient given, nothing was sent.")
        return 1 if args.strict else 0

    if not fc.credentials_present():
        print("[WARN] FEISHU_APP_ID / FEISHU_APP_SECRET not set, nothing was sent.")
        return 1 if args.strict else 0

    try:
        token = fc.get_token()
    except fc.FeishuError as e:
        print(f"[ERROR] Could not authenticate: {e}")
        return 1 if args.strict else 0

    # One recipient failing must not cost the others their message, so each
    # send is reported on its own and only the tally decides the exit code.
    sent, failed = 0, 0
    for recipient in recipients:
        try:
            message_id = fc.send_message(token, recipient, card, receive_id_type=receive_id_type)
        except fc.FeishuError as e:
            # Feishu answers every routing problem with the same permission
            # error, so the likely cause has to be named rather than read off
            # the response.
            print(f"[ERROR] Sending to {recipient} failed: {e}")
            if receive_id_type == "open_id":
                print("[HINT] If this is a permission error, check the app's "
                      "availability scope in the developer console: a person "
                      "outside it cannot be messaged. Note also that an "
                      "open_id is scoped to one app, so an id resolved through "
                      "a different app is accepted here and never delivers.")
            else:
                print("[HINT] If this is a permission error, check that the bot "
                      "was added to the chat: chat settings -> Group Bots -> "
                      "Add Bot.")
            failed += 1
            continue
        print(f"[INFO] Sent message {message_id} to {recipient}")
        sent += 1

    if failed and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
