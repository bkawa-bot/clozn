"""wire_denoise.py -- inject the real denoise_trace.json into inspector/demo/denoise.html (window.DENOISE).
Run: C:/Users/brigi/src/clozn/.venv-sae/Scripts/python.exe research/wire_denoise.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "inspector", "demo")
DATA = os.path.join(DEMO, "denoise_trace.json")
HTML = os.path.join(DEMO, "denoise.html")
START, END = "<!-- DENOISE-INJECT-START -->", "<!-- DENOISE-INJECT-END -->"


def main():
    data = json.load(open(DATA, encoding="utf-8"))
    block = f"{START}\n<script>window.DENOISE = {json.dumps(data, ensure_ascii=False)};</script>\n{END}"
    html = re.sub(re.escape(START) + r".*?" + re.escape(END), block,
                  open(HTML, encoding="utf-8").read(), flags=re.S)
    open(HTML, "w", encoding="utf-8").write(html)
    print(f"wired {len(data.get('passes', []))} passes into {os.path.normpath(HTML)}")


if __name__ == "__main__":
    main()
