"""
Knowledge-graph view (wireframe 1c).

Nodes are laid out at their real position on the sheet rather than by a force
simulation. For a P&ID that is strictly better: the topology IS spatial, so a
force layout would scramble the one thing an engineer already knows how to read.
Zoom, pan, click to trace, and filter by category.
"""
from __future__ import annotations
import json
from pathlib import Path
import yaml

_ONT = yaml.safe_load(open(Path(__file__).resolve().parents[3] / "config" / "symbol_ontology.yaml"))
CLASSES = _ONT["classes"]
CAT_COLOR = {
    "valve": "#5aa9ff", "instrument": "#ff9d5c", "fitting": "#7f8794",
    "actuator": "#c78bff", "equipment": "#4fd08a", "annotation": "#e3c05a",
    "line": "#00e5ff", "connector": "#ff5252", "unknown": "#7f8794",
}


def render(entities: list[dict], relations: list[dict], width: int, height: int,
           focus: str | None = None, path_nodes: list[str] | None = None,
           height_px: int = 620, derived: dict | None = None) -> str:
    scale = 1400 / max(width, 1)
    vw, vh = int(width * scale), int(height * scale)
    pathset = set(path_nodes or [])

    nodes = []
    for e in entities:
        x0, y0, x1, y1 = e["bbox"]
        cat = e.get("category") or CLASSES.get(e["pidqa_class"], {}).get("category", "fitting")
        name = e.get("name") or CLASSES.get(e["pidqa_class"], {}).get("name", f"class {e['pidqa_class']}")
        nodes.append({
            "id": e["id"],
            "x": round((x0 + x1) / 2 * scale, 1), "y": round((y0 + y1) / 2 * scale, 1),
            "t": e.get("tag") or "", "c": e["pidqa_class"], "cat": cat,
            "n": name,
            "col": CAT_COLOR.get(cat, "#7f8794"),
            "p": e["id"] in pathset,
        })
    pos = {n["id"]: n for n in nodes}
    edges = [{"a": r["source"], "b": r["target"], "t": "c"}
             for r in relations if r["source"] in pos and r["target"] in pos]
    for key, code in (("CONTROLS", "k"), ("FEEDS_TO", "f")):
        edges += [{"a": r["source"], "b": r["target"], "t": code}
                  for r in (derived or {}).get(key, [])
                  if r["source"] in pos and r["target"] in pos]
    payload = json.dumps({"nodes": nodes, "edges": edges, "vw": vw, "vh": vh, "focus": focus})

    return f"""
<style>
 .g-wrap{{position:relative;height:{height_px}px;border-radius:10px;overflow:hidden;
   background:radial-gradient(ellipse at 50% 0%,#12151b 0%,#0a0b0e 70%);
   border:1px solid rgba(255,255,255,.09);
   font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}}
 .g-stage{{position:absolute;inset:0;cursor:grab;overflow:hidden;}}
 .g-stage.drag{{cursor:grabbing;}}
 .g-inner{{position:absolute;transform-origin:0 0;}}
 .g-bar{{position:absolute;left:12px;bottom:12px;display:flex;gap:6px;z-index:5;flex-wrap:wrap;}}
 .g-chip{{background:rgba(20,22,27,.9);border:1px solid rgba(255,255,255,.12);color:#c6cad2;
   border-radius:20px;padding:4px 11px;font-size:11px;cursor:pointer;transition:.14s;
   display:flex;align-items:center;gap:5px;}}
 .g-chip.off{{opacity:.35;}}
 .g-chip i{{width:7px;height:7px;border-radius:50%;display:inline-block;}}
 .g-hud{{position:absolute;right:12px;bottom:12px;z-index:5;font:500 11px ui-monospace,Menlo,monospace;
   color:#7d828d;background:rgba(14,15,19,.86);padding:5px 10px;border-radius:7px;
   border:1px solid rgba(255,255,255,.07);}}
 .g-card{{position:absolute;right:12px;top:12px;width:230px;z-index:6;display:none;
   background:rgba(16,18,23,.96);border:1px solid rgba(255,255,255,.11);border-radius:10px;
   padding:13px 14px;backdrop-filter:blur(14px);box-shadow:0 10px 34px rgba(0,0,0,.5);}}
 .g-card h4{{margin:0 0 2px;font-size:13px;color:#f0f1f4;font-weight:600;}}
 .g-card .tg{{font:600 12px ui-monospace,Menlo,monospace;color:#5aa9ff;margin-bottom:8px;}}
 .g-card .nb{{font-size:11px;color:#9aa0ab;line-height:1.7;max-height:150px;overflow:auto;}}
 .g-card .nb b{{color:#dfe2e8;font-family:ui-monospace,Menlo,monospace;font-weight:500;}}
 circle.nd{{cursor:pointer;stroke-width:1.4;vector-effect:non-scaling-stroke;}}
 text.gl{{dominant-baseline:middle;}}
 circle.nd:hover{{stroke:#fff;stroke-width:2.4;}}
 circle.nd.on{{stroke:#fff;stroke-width:2.6;}}
 circle.nd.path{{stroke:#5aa9ff;stroke-width:3;}}
 line.ed{{stroke:rgba(255,255,255,.16);}}
 line.ed.k{{stroke:rgba(79,208,138,.75);stroke-dasharray:5 4;}}
 line.ed.f{{stroke:rgba(227,192,90,.8);}}
 line.ed.on{{stroke:rgba(90,169,255,.85);}}
 text.gl{{fill:#aeb4bf;font:500 8px ui-monospace,Menlo,monospace;pointer-events:none;
   paint-order:stroke;stroke:rgba(8,9,12,.9);stroke-width:2.4px;}}
</style>
<div class="g-wrap">
  <div class="g-stage" id="gs"><div class="g-inner" id="gi">
    <svg id="gsvg" width="{vw}" height="{vh}"></svg>
  </div></div>
  <div class="g-card" id="gc"></div>
  <div class="g-bar" id="gb"></div>
  <div class="g-hud" id="gh"></div>
</div>
<script>
(function(){{
  const D={payload}, NS='http://www.w3.org/2000/svg';
  const gs=document.getElementById('gs'), gi=document.getElementById('gi'),
        sv=document.getElementById('gsvg'), gc=document.getElementById('gc'),
        gh=document.getElementById('gh'), gb=document.getElementById('gb');
  const byId={{}}; D.nodes.forEach(n=>byId[n.id]=n);
  const adj={{}}; D.edges.forEach(e=>{{(adj[e.a]=adj[e.a]||[]).push(e.b);(adj[e.b]=adj[e.b]||[]).push(e.a);}});
  const cats=[...new Set(D.nodes.map(n=>n.cat))];
  const off=new Set(); let sel=D.focus||null;

  gb.innerHTML=cats.map(c=>{{
    const col=D.nodes.find(n=>n.cat===c).col, k=D.nodes.filter(n=>n.cat===c).length;
    return `<div class="g-chip" data-c="${{c}}"><i style="background:${{col}}"></i>${{c}} ${{k}}</div>`;}}).join('');
  gb.querySelectorAll('.g-chip').forEach(el=>el.onclick=()=>{{
    const c=el.dataset.c; if(off.has(c)){{off.delete(c);el.classList.remove('off');}}
    else{{off.add(c);el.classList.add('off');}} draw();}});

  function vis(n){{return !off.has(n.cat);}}
  function draw(){{
    sv.innerHTML='';
    const near=sel?new Set([sel,...(adj[sel]||[])]):null;
    D.edges.forEach(e=>{{
      const a=byId[e.a],b=byId[e.b]; if(!a||!b||!vis(a)||!vis(b))return;
      const l=document.createElementNS(NS,'line');
      l.setAttribute('x1',a.x);l.setAttribute('y1',a.y);
      l.setAttribute('x2',b.x);l.setAttribute('y2',b.y);
      const on=sel&&(e.a===sel||e.b===sel);
      l.setAttribute('class','ed '+(e.t==='k'?'k':(e.t==='f'?'f':''))+(on?' on':'')); sv.appendChild(l);
    }});
    let shown=0;
    D.nodes.forEach(n=>{{
      if(!vis(n))return; shown++;
      const c=document.createElementNS(NS,'circle');
      c.setAttribute('cx',n.x);c.setAttribute('cy',n.y);
      c.setAttribute('r', n.p?6.2/z:4.6/z);
      c.setAttribute('fill',n.col);
      c.setAttribute('stroke','rgba(0,0,0,.55)');
      c.setAttribute('class','nd'+(n.id===sel?' on':'')+(n.p?' path':''));
      c.onclick=ev=>{{ev.stopPropagation();pick(n.id);}};
      sv.appendChild(c);
      if(n.t&&(!sel||near.has(n.id))){{
        const t=document.createElementNS(NS,'text');
        t.setAttribute('x',n.x+7);t.setAttribute('y',n.y+3);
        t.dataset.x=n.x; t.dataset.y=n.y;
        t.setAttribute('class','gl');t.textContent=n.t.slice(0,14);sv.appendChild(t);
      }}
    }});
    gh.textContent=`${{shown}} nodes · ${{D.edges.length}} edges`;
    rescale();
  }}
  function pick(id){{
    sel=id;draw();const n=byId[id];
    const nb=(adj[id]||[]).slice(0,9).map(x=>{{const o=byId[x];
      return `<div>→ <b>${{o.t||'untagged'}}</b> <span style="opacity:.6">${{o.n}}</span></div>`;}}).join('')
      ||'<div style="opacity:.55">no connections</div>';
    gc.innerHTML=`<h4>${{n.n}}</h4><div class="tg">${{n.t||'— untagged —'}}</div>
      <div class="nb"><div style="margin-bottom:4px;opacity:.7">connected_to (${{(adj[id]||[]).length}})</div>${{nb}}</div>`;
    gc.style.display='block';
  }}
  let z=1,ox=0,oy=0;
  // SVG radii and font sizes scale with the transform, so at fit zoom (~0.26 on
  // a 7168px sheet) a 4px node renders as 1px and disappears. Counter-scale them
  // so nodes keep a constant on-screen size at any zoom.
  function rescale(){{
    const r=4.6/z, rp=6.2/z, fs=Math.max(3,9/z);
    sv.querySelectorAll('circle.nd').forEach(c=>
      c.setAttribute('r', c.classList.contains('path')?rp:r));
    sv.querySelectorAll('text.gl').forEach(t=>{{
      t.setAttribute('font-size',fs);
      t.setAttribute('x',(+t.dataset.x)+r*1.7);
      t.setAttribute('y',(+t.dataset.y)+fs*0.34);
      t.setAttribute('stroke-width',2.6/z);
    }});
    sv.querySelectorAll('line.ed').forEach(l=>
      l.setAttribute('stroke-width',(l.classList.contains('on')?2.2:1)/z));
  }}
  const ap=()=>{{gi.style.transform=`translate(${{ox}}px,${{oy}}px) scale(${{z}})`;rescale();}};
  function fit(){{
    const r=gs.getBoundingClientRect();
    if(!r.width || !r.height || r.width <= 10 || r.height <= 10) return;
    z=Math.min(r.width/D.vw, r.height/D.vh)*0.93;
    if(!z || !isFinite(z) || z <= 0) z = 0.5;
    ox=(r.width-D.vw*z)/2; oy=(r.height-D.vh*z)/2;
    ap();
  }}
  gs.addEventListener('wheel',e=>{{e.preventDefault();
    const r=gs.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
    const f=e.deltaY<0?1.14:1/1.14,nz=Math.max(.1,Math.min(12,z*f));
    ox=mx-(mx-ox)*(nz/z);oy=my-(my-oy)*(nz/z);z=nz;ap();}},{{passive:false}});
  let dr=false,px=0,py=0;
  gs.addEventListener('mousedown',e=>{{dr=true;px=e.clientX;py=e.clientY;gs.classList.add('drag');}});
  window.addEventListener('mouseup',()=>{{dr=false;gs.classList.remove('drag');}});
  window.addEventListener('mousemove',e=>{{if(!dr)return;ox+=e.clientX-px;oy+=e.clientY-py;px=e.clientX;py=e.clientY;ap();}});
  gs.addEventListener('click',e=>{{if(e.target===gs||e.target===sv){{sel=null;gc.style.display='none';draw();}}}});
  draw();
  fit();
  if(window.ResizeObserver){{
    const ro = new ResizeObserver(entries => {{
      for(const entry of entries){{
        if(entry.contentRect.width > 20 && entry.contentRect.height > 20){{
          fit();
        }}
      }}
    }});
    ro.observe(gs);
  }}
  let fitTries = 0;
  const checkTimer = setInterval(() => {{
    fitTries++;
    const r = gs.getBoundingClientRect();
    if(r.width > 20 && r.height > 20){{
      fit();
      if(fitTries > 5) clearInterval(checkTimer);
    }}
    if(fitTries > 30) clearInterval(checkTimer);
  }}, 250);
  if(D.focus) pick(D.focus);
}})();
</script>"""
