"""
Interactive drawing canvas.

A P&ID is 7168x4562 px. A static downscaled <img> is useless for inspection -
the reviewer has to be able to zoom to a symbol and read its tag. This renders a
self-contained HTML/SVG component with wheel zoom, drag pan, click-to-select and
an in-canvas detail card. No external JS: it must work offline.

The base raster is embedded as a data URI so the component has no server
dependency, downscaled to keep the payload sane while staying sharp at 4x zoom.
"""
from __future__ import annotations
import base64, json
from pathlib import Path
import cv2
import numpy as np
import yaml

_ONT = yaml.safe_load(open(Path(__file__).resolve().parents[3] / "config" / "symbol_ontology.yaml"))
CLASSES = _ONT["classes"]

CAT_COLOR = {
    "valve":      "#5aa9ff",
    "instrument": "#ff9d5c",
    "fitting":    "#7f8794",
    "actuator":   "#c78bff",
    "equipment":  "#4fd08a",
    "annotation": "#e3c05a",
}
BASE_WIDTH = 2400


def _data_uri(img: np.ndarray, width: int = BASE_WIDTH, quality: int = 84) -> tuple[str, float]:
    h, w = img.shape[:2]
    scale = min(1.0, width / w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(), scale


def render(img: np.ndarray, entities: list[dict], relations: list[dict],
           highlight: list[str] | None = None, selected: str | None = None,
           show_links: bool = False, show_labels: bool = True,
           height: int = 620, derived: dict | None = None) -> str:
    uri, scale = _data_uri(img)
    H, W = img.shape[:2]
    vw, vh = int(W * scale), int(H * scale)
    hi = set(highlight or [])

    nodes = []
    for e in entities:
        x0, y0, x1, y1 = [v * scale for v in e["bbox"]]
        cat = CLASSES.get(e["pidqa_class"], {}).get("category", "fitting")
        nodes.append({
            "id": e["id"], "x": round(x0, 1), "y": round(y0, 1),
            "w": round(x1 - x0, 1), "h": round(y1 - y0, 1),
            "cx": round((x0 + x1) / 2, 1), "cy": round((y0 + y1) / 2, 1),
            "c": e["pidqa_class"],
            "n": CLASSES.get(e["pidqa_class"], {}).get("name", f"class {e['pidqa_class']}"),
            "t": e.get("tag") or "", "tc": round(float(e.get("tag_confidence", 0)), 2),
            "dc": round(float(e.get("confidence", 1.0)), 2),
            "col": CAT_COLOR.get(cat, "#7f8794"),
            "hi": e["id"] in hi,
            "bbox": e["bbox"],
        })
    pos = {n["id"]: (n["cx"], n["cy"]) for n in nodes}
    edges = [{"x1": pos[r["source"]][0], "y1": pos[r["source"]][1],
              "x2": pos[r["target"]][0], "y2": pos[r["target"]][1]}
             for r in relations if r["source"] in pos and r["target"] in pos]

    def _typed(key):
        return [{"x1": pos[r["source"]][0], "y1": pos[r["source"]][1],
                 "x2": pos[r["target"]][0], "y2": pos[r["target"]][1]}
                for r in (derived or {}).get(key, [])
                if r["source"] in pos and r["target"] in pos]

    payload = json.dumps({"nodes": nodes, "edges": edges, "vw": vw, "vh": vh,
                          "selected": selected, "links": show_links,
                          "labels": show_labels,
                          "controls": _typed("CONTROLS"), "feeds": _typed("FEEDS_TO")})

    return f"""
<style>
 html,body{{background:transparent;margin:0;}}
 /* the viewport: a bracketed, scanlined instrument frame */
 .edi-wrap{{position:relative;height:{height}px;overflow:hidden;
   background:
     linear-gradient(rgba(226,232,240,.7),rgba(226,232,240,.7)) 0 0/13px 2px no-repeat,
     linear-gradient(rgba(226,232,240,.7),rgba(226,232,240,.7)) 0 0/2px 13px no-repeat,
     linear-gradient(rgba(226,232,240,.7),rgba(226,232,240,.7)) 100% 0/13px 2px no-repeat,
     linear-gradient(rgba(226,232,240,.7),rgba(226,232,240,.7)) 100% 0/2px 13px no-repeat,
     linear-gradient(rgba(226,232,240,.7),rgba(226,232,240,.7)) 0 100%/13px 2px no-repeat,
     linear-gradient(rgba(226,232,240,.7),rgba(226,232,240,.7)) 0 100%/2px 13px no-repeat,
     linear-gradient(rgba(226,232,240,.7),rgba(226,232,240,.7)) 100% 100%/13px 2px no-repeat,
     linear-gradient(rgba(226,232,240,.7),rgba(226,232,240,.7)) 100% 100%/2px 13px no-repeat,
     rgba(0,0,0,.55);
   border:1px solid rgba(226,232,240,.28);
   box-shadow:inset 0 0 20px rgba(226,232,240,.05);
   font-family:'Space Mono','JetBrains Mono',ui-monospace,Menlo,monospace;}}
 /* No scanline wash over the sheet itself. The reference lays it over dark
    telemetry panels where it reads as CRT texture; over a white P&ID it is
    stripes across the thing the user came to read. */
 .edi-stage{{position:absolute;inset:0;cursor:grab;overflow:hidden;
   box-shadow:inset 0 1px 2px rgba(0,0,0,.5);}}
 .edi-stage.drag{{cursor:grabbing;}}
 .edi-inner{{position:absolute;transform-origin:0 0;will-change:transform;}}
 /* a scanned sheet is pure white and glares on a near-black console. Easing it
    back a little keeps every line readable while letting it sit IN the panel
    rather than on top of it. */
 .edi-inner img{{display:block;image-rendering:auto;
   filter:brightness(.90) contrast(1.06);}}
 .edi-inner svg{{position:absolute;left:0;top:0;overflow:visible;}}
 .edi-bar{{position:absolute;left:12px;bottom:12px;display:flex;gap:6px;z-index:5;}}
 .edi-btn{{background:rgba(0,0,0,.6);border:1px solid rgba(226,232,240,.26);
   color:rgba(226,232,240,.8);border-radius:0;width:28px;height:28px;font-size:13px;
   cursor:pointer;display:grid;place-items:center;transition:.15s;}}
 .edi-btn:hover{{background:rgba(226,232,240,.10);border-color:rgba(226,232,240,.7);}}
 .edi-btn.w{{width:auto;padding:0 11px;font-size:11.5px;letter-spacing:.02em;}}
 .edi-hud{{position:absolute;right:12px;bottom:12px;z-index:5;font:500 11px ui-monospace,Menlo,monospace;
   color:rgba(226,232,240,.55);background:rgba(0,0,0,.6);padding:5px 11px;
   border-radius:0;border:1px solid rgba(226,232,240,.2);
   letter-spacing:.14em;text-transform:uppercase;}}
 .edi-card{{position:absolute;right:12px;top:12px;width:236px;z-index:6;
   background:rgba(0,0,0,.9);border:1px solid rgba(226,232,240,.32);border-radius:0;
   padding:13px 14px;backdrop-filter:blur(14px);display:none;
   box-shadow:0 10px 34px rgba(0,0,0,.5);}}
 .edi-card h4{{margin:0 0 3px;font-size:13.5px;color:#f0f1f4;font-weight:600;letter-spacing:-.01em;}}
 .edi-card .tag{{font:600 12px ui-monospace,Menlo,monospace;color:#5aa9ff;margin-bottom:9px;}}
 .edi-card dl{{margin:0;display:grid;grid-template-columns:auto 1fr;gap:3px 10px;
   font-size:11.5px;color:#9aa0ab;}}
 .edi-card dd{{margin:0;color:#dfe2e8;font-family:ui-monospace,Menlo,monospace;text-align:right;}}
 .edi-card .nb{{margin-top:9px;padding-top:8px;border-top:1px solid rgba(255,255,255,.09);
   font-size:11px;color:#9aa0ab;line-height:1.65;max-height:118px;overflow:auto;}}
 .edi-card .nb b{{color:#dfe2e8;font-family:ui-monospace,Menlo,monospace;font-weight:500;}}
 .edi-legend{{position:absolute;left:12px;top:12px;z-index:5;display:flex;flex-wrap:wrap;
   gap:4px 9px;max-width:60%;font-size:10.5px;color:#8b909b;
   background:rgba(0,0,0,.7);padding:6px 10px;border-radius:0;
   border:1px solid rgba(226,232,240,.18);letter-spacing:.1em;}}
 .edi-legend i{{width:8px;height:8px;border-radius:2px;display:inline-block;margin-right:4px;}}
 rect.sym{{fill:transparent;stroke-width:1.4;vector-effect:non-scaling-stroke;cursor:pointer;}}
 rect.sym:hover{{stroke-width:3;}}
 rect.sym.hi{{stroke-width:3;fill:rgba(90,169,255,.16);}}
 rect.sym.sel{{stroke:#fff!important;stroke-width:3.4;fill:rgba(255,255,255,.13);}}
 line.rel{{stroke:rgba(127,135,148,.45);stroke-width:1;vector-effect:non-scaling-stroke;}}
 line.ctl{{stroke:rgba(79,208,138,.8);stroke-width:1.6;vector-effect:non-scaling-stroke;
   stroke-dasharray:5 4;}}
 line.fed{{stroke:rgba(227,192,90,.85);stroke-width:1.8;vector-effect:non-scaling-stroke;}}
 text.lbl{{fill:#c9ced8;font:500 9px ui-monospace,Menlo,monospace;pointer-events:none;
   paint-order:stroke;stroke:rgba(8,9,12,.85);stroke-width:2.6px;}}
</style>

<div class="edi-wrap">
  <div class="edi-stage" id="stage">
    <div class="edi-inner" id="inner">
      <img src="{uri}" width="{vw}" height="{vh}" draggable="false">
      <svg id="ov" width="{vw}" height="{vh}"></svg>
    </div>
  </div>
  <div class="edi-legend" id="legend"></div>
  <div class="edi-card" id="card"></div>
  <div class="edi-bar">
    <button class="edi-btn" id="zin">+</button>
    <button class="edi-btn" id="zout">&minus;</button>
    <button class="edi-btn w" id="fit">fit</button>
    <button class="edi-btn w" id="tgl">links</button>
    <button class="edi-btn w" id="tctl">controls</button>
  </div>
  <div class="edi-hud" id="hud"></div>
</div>

<script>
(function(){{
  const D = {payload};
  const stage=document.getElementById('stage'), inner=document.getElementById('inner'),
        ov=document.getElementById('ov'), card=document.getElementById('card'),
        hud=document.getElementById('hud'), legend=document.getElementById('legend');
  const NS='http://www.w3.org/2000/svg';
  const byId={{}}; D.nodes.forEach(n=>byId[n.id]=n);
  const nbrs={{}};
  D.edges.forEach(()=>{{}});

  // adjacency from edge endpoints (matched back by centre coords)
  const key=(x,y)=>x+','+y, cmap={{}};
  D.nodes.forEach(n=>cmap[key(n.cx,n.cy)]=n.id);
  D.edges.forEach(e=>{{
    const a=cmap[key(e.x1,e.y1)], b=cmap[key(e.x2,e.y2)];
    if(a&&b){{ (nbrs[a]=nbrs[a]||[]).push(b); (nbrs[b]=nbrs[b]||[]).push(a); }}
  }});

  let showLinks=D.links, showCtl=false;
  function draw(){{
    ov.innerHTML='';
    const drawSet=(arr,cls)=>arr.forEach(e=>{{
      const l=document.createElementNS(NS,'line');
      l.setAttribute('x1',e.x1);l.setAttribute('y1',e.y1);
      l.setAttribute('x2',e.x2);l.setAttribute('y2',e.y2);
      l.setAttribute('class',cls); ov.appendChild(l);
    }});
    if(showLinks) drawSet(D.edges,'rel');
    if(showCtl){{ drawSet(D.feeds,'fed'); drawSet(D.controls,'ctl'); }}
    D.nodes.forEach(n=>{{
      const r=document.createElementNS(NS,'rect');
      r.setAttribute('x',n.x);r.setAttribute('y',n.y);
      r.setAttribute('width',Math.max(n.w,3));r.setAttribute('height',Math.max(n.h,3));
      r.setAttribute('stroke',n.col);
      r.setAttribute('class','sym'+(n.hi?' hi':'')+(n.id===D.selected?' sel':''));
      r.dataset.id=n.id;
      r.addEventListener('click',ev=>{{ev.stopPropagation();select(n.id);}});
      ov.appendChild(r);
      if(D.labels && n.t){{
        const t=document.createElementNS(NS,'text');
        t.setAttribute('x',n.x);t.setAttribute('y',n.y-3);
        t.setAttribute('class','lbl');t.textContent=n.t.slice(0,16);
        ov.appendChild(t);
      }}
    }});
  }}

  const cats={{}};
  D.nodes.forEach(n=>cats[n.col]=(cats[n.col]||0)+1);
  legend.innerHTML=Object.entries({{'#5aa9ff':'valve','#ff9d5c':'instrument','#7f8794':'fitting',
    '#c78bff':'actuator','#4fd08a':'equipment','#e3c05a':'annotation'}})
    .filter(([c])=>cats[c]).map(([c,l])=>`<span><i style="background:${{c}}"></i>${{l}} ${{cats[c]}}</span>`).join('')
    + (D.controls.length?`<span style="color:#4fd08a">— — controls ${{D.controls.length}}</span>`:'')
    + (D.feeds.length?`<span style="color:#e3c05a">&#8594; feeds_to ${{D.feeds.length}}</span>`:'');

  function select(id){{
    D.selected=id; draw();
    const n=byId[id]; if(!n){{card.style.display='none';return;}}
    const nb=(nbrs[id]||[]).slice(0,7).map(x=>{{const o=byId[x];
      return `<div>→ <b>${{o.t||'untagged'}}</b> <span style="opacity:.62">${{o.n}}</span></div>`;}}).join('')
      || '<div style="opacity:.55">no connections traced</div>';
    card.innerHTML=`<h4>${{n.n}}</h4><div class="tag">${{n.t||'— untagged —'}}</div>
      <dl><dt>class</dt><dd>${{n.c}}</dd><dt>detect</dt><dd>${{n.dc.toFixed(2)}}</dd>
      <dt>tag conf</dt><dd>${{n.tc.toFixed(2)}}</dd>
      <dt>bbox</dt><dd style="font-size:10px">${{n.bbox.join(', ')}}</dd></dl>
      <div class="nb"><div style="margin-bottom:4px;opacity:.72">connected_to</div>${{nb}}</div>`;
    card.style.display='block';
  }}

  let z=1,ox=0,oy=0;
  function apply(){{
    inner.style.transform=`translate(${{ox}}px,${{oy}}px) scale(${{z}})`;
    hud.textContent=`${{Math.round(z*100)}}% · ${{D.nodes.length}} entities · ${{D.edges.length}} relations`;
  }}
  function fit(){{
    const r=stage.getBoundingClientRect();
    z=Math.min(r.width/D.vw, r.height/D.vh)*0.97;
    ox=(r.width-D.vw*z)/2; oy=(r.height-D.vh*z)/2; apply();
  }}
  stage.addEventListener('wheel',e=>{{
    e.preventDefault();
    const r=stage.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
    const f=e.deltaY<0?1.14:1/1.14, nz=Math.max(0.05,Math.min(14,z*f));
    ox=mx-(mx-ox)*(nz/z); oy=my-(my-oy)*(nz/z); z=nz; apply();
  }},{{passive:false}});
  let dr=false,px=0,py=0;
  stage.addEventListener('mousedown',e=>{{dr=true;px=e.clientX;py=e.clientY;stage.classList.add('drag');}});
  window.addEventListener('mouseup',()=>{{dr=false;stage.classList.remove('drag');}});
  window.addEventListener('mousemove',e=>{{if(!dr)return;ox+=e.clientX-px;oy+=e.clientY-py;px=e.clientX;py=e.clientY;apply();}});
  stage.addEventListener('click',e=>{{if(e.target===stage||e.target.tagName==='IMG'){{card.style.display='none';D.selected=null;draw();}}}});
  document.getElementById('zin').onclick=()=>{{z=Math.min(14,z*1.35);apply();}};
  document.getElementById('zout').onclick=()=>{{z=Math.max(0.05,z/1.35);apply();}};
  document.getElementById('fit').onclick=fit;
  document.getElementById('tgl').onclick=()=>{{showLinks=!showLinks;draw();}};
  document.getElementById('tctl').onclick=()=>{{showCtl=!showCtl;draw();}};

  draw(); fit();
  if(D.selected) select(D.selected);
  // frame the highlighted evidence if there is any
  const h=D.nodes.filter(n=>n.hi);
  if(h.length){{
    const x0=Math.min(...h.map(n=>n.x)), y0=Math.min(...h.map(n=>n.y)),
          x1=Math.max(...h.map(n=>n.x+n.w)), y1=Math.max(...h.map(n=>n.y+n.h));
    const r=stage.getBoundingClientRect(), pad=170;
    z=Math.max(0.15,Math.min(7,Math.min(r.width/(x1-x0+pad), r.height/(y1-y0+pad))));
    ox=r.width/2-((x0+x1)/2)*z; oy=r.height/2-((y0+y1)/2)*z; apply();
  }}
}})();
</script>"""
