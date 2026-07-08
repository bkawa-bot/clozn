"""wire_memory.py -- inject the real memory_timeline.json into inspector/demo/memory.html.

Same pattern as wire_atlas: the viz reads window.MEMORY; this replaces the placeholder block between the
MEMORY-INJECT markers with the actual run, so memory.html opens straight from disk showing real data.

Run: C:/Users/brigi/src/clozn/.venv-sae/Scripts/python.exe research/wire_memory.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "inspector", "demo")
DATA = os.path.join(DEMO, "memory_timeline.json")
HTML = os.path.join(DEMO, "memory.html")
START, END = "<!-- MEMORY-INJECT-START -->", "<!-- MEMORY-INJECT-END -->"


def main():
    data = json.load(open(DATA, encoding="utf-8"))
    block = f"{START}\n<script>window.MEMORY = {json.dumps(data, ensure_ascii=False)};</script>\n{END}"
    html = open(HTML, encoding="utf-8").read()
    html = re.sub(re.escape(START) + r".*?" + re.escape(END), block, html, flags=re.S)
    open(HTML, "w", encoding="utf-8").write(html)
    print(f"wired {len(data.get('turns', []))} turns / {len(data.get('checkpoints', []))} checkpoints "
          f"into {os.path.normpath(HTML)}")


if __name__ == "__main__":
    main()
