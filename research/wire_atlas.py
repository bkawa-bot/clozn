"""wire_atlas.py -- inject the REAL concept atlas into the brain visualization.

The viz (inspector/demo/brain.html) reads window.BRAIN if present, else an inline synthetic seed.
This transforms inspector/demo/atlas.json (the verified SAE-feature atlas from feature_atlas.py) into
the viz's schema {clusters:[{id,label,color}], nodes:[{id,label,cluster,value}], links:[{source,target,
weight}]} and inlines it as a window.BRAIN <script> before the renderer -- so brain.html opens straight
from disk showing the REAL feature atlas (file:// blocks fetch, hence inlining). Idempotent.

Run: C:/Users/brigi/src/clozn/.venv-sae/Scripts/python.exe research/wire_atlas.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "..", "inspector", "demo")
ATLAS = os.path.join(DEMO, os.environ.get("ATLAS_JSON", "atlas7b.json"))   # 7B atlas by default
HTML = os.path.join(DEMO, "brain.html")
START, END = "<!-- REAL-ATLAS-INJECT-START -->", "<!-- REAL-ATLAS-INJECT-END -->"

# one luminous color per concept (the light "Artificial Angels" family; the viz re-luminizes for theme)
COLORS = ["#36AEC4", "#E89BB0", "#9B8CE8", "#3FC4A8", "#E8C36B", "#E89A55", "#7FC46B", "#E8806B",
          "#6B9BE8", "#5FD9B3", "#8CA8E8", "#52C6E8", "#B89BE8", "#7B86D8", "#E87BB0",
          "#5FB0E8", "#3FB8C4", "#C4836B", "#6BD0B8", "#D9B84F", "#9BD06B", "#C49BE8"]


def main():
    atlas = json.load(open(ATLAS, encoding="utf-8"))
    concepts = atlas["meta"]["concepts"]
    clusters = [{"id": concepts[i], "label": concepts[i].capitalize(), "color": COLORS[i % len(COLORS)]}
                for i in range(len(concepts))]
    nodes = [{"id": n["id"], "label": n["label"], "cluster": concepts[n["cluster"]], "value": n["value"]}
             for n in atlas["nodes"]]
    links = [{"source": l["source"], "target": l["target"], "weight": l.get("weight", 1)}
             for l in atlas["links"]]
    brain = {"clusters": clusters, "nodes": nodes, "links": links}

    inject = (f"{START}\n<script>window.BRAIN = {json.dumps(brain, ensure_ascii=False)};</script>\n{END}\n")
    html = open(HTML, encoding="utf-8").read()
    html = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n?", "", html, flags=re.S)  # idempotent
    idx = html.find('<script type="module"')
    if idx < 0:
        raise SystemExit("could not find the renderer <script type=module> in brain.html")
    html = html[:idx] + inject + html[idx:]
    with open(HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wired {len(nodes)} real feature nodes / {len(links)} links / {len(clusters)} concept lobes "
          f"into {os.path.normpath(HTML)}")
    print("concept lobes:", ", ".join(concepts))


if __name__ == "__main__":
    main()
