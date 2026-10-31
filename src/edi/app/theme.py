"""
Visual system — spacecraft cockpit wireframe.

Rebuilt against the shared reference (Spacecraft Cockpit Wireframe UI). Read off
its stylesheet rather than guessed at:

    body        Space Mono, monospace — EVERYTHING is mono. That one decision
                does more than any other to make it read as instrumentation
    stage       rgb(3,3,3)
    foreground  rgb(226,232,240) — slate, not pure white
    panel       rgba(0,0,0,.6) · 1px solid rgba(226,232,240,.4) · blur(8px)
                · inset 0 0 20px rgba(226,232,240,.05)
    corners     four 10px L-brackets, 2px, on every panel
    scanlines   a 4px repeating horizontal wash at rgba(0,0,0,.2)
    grid        50px blueprint grid at rgba(226,232,240,.05)
    header      10px, tracking-widest, a bottom hairline, label left and status
                right at 50% opacity
    rows        `LABEL:` dim, value bold, on one line
    strokes     1.5px solid · 0.5px at 50% · 1px dashed 4,4
    radius      NONE. Every corner is square.

Two deliberate departures:

* **Corner brackets are drawn with background layers**, not four child divs.
  Streamlit owns the DOM; it does not own the CSS.
* **The four indicator colours survive** as 5px lamps. The reference is a
  fictional cockpit and can be strictly monochrome; this screen has to tell a
  technician that a line is flammable, and hue is the fastest channel there is.
  Nothing else in the interface is coloured.
"""

# Drop this as the FIRST thing inside `st.container(border=True)` to turn it
# into a HUD panel. It renders nothing.
PANEL = '<span class="pnl-mark"></span>'


def hud(label: str, status: str = "") -> str:
    """A panel header: system name on the left, state on the right."""
    right = f'<span class="st">{status}</span>' if status else ""
    return f'<div class="hud-hd"><span>{label}</span>{right}</div>'


def telemetry(cells: list[tuple[str, str]], seq: str = "") -> str:
    """The fixed status strip along the bottom of the console.

    The reference carries one — `SEQ: 0x99A4F // PROTOCOL: OMEGA` — and it is a
    large part of why it reads as a machine rather than a web page. Here it
    carries things that are actually true: which sheet is loaded, how big its
    graph is, and whether anything left the building.
    """
    body = "".join(f'<span class="tl-c"><i>{k}</i>{v}</span>' for k, v in cells)
    tail = f'<span class="tl-seq">{seq}</span>' if seq else ""
    return f'<div class="telem"><div class="tl-in">{body}{tail}</div></div>'


def seg(filled: int, total: int = 8) -> str:
    """A segment meter, as the reference draws a level."""
    cells = "".join(f'<i class="{"on" if i < filled else ""}"></i>'
                    for i in range(total))
    return f'<span class="seg">{cells}</span>'


CSS = """
<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

 :root{
   --bg:#030303; --fg:#e2e8f0;
   --l1:rgba(226,232,240,.14);   /* hairline, structural */
   --l2:rgba(226,232,240,.26);   /* control borders */
   --l3:rgba(226,232,240,.42);   /* panel edge */
   --l4:rgba(226,232,240,.70);   /* emphasis */
   --tx:rgba(226,232,240,.96);
   --txb:rgba(226,232,240,.78);
   --tx2:rgba(226,232,240,.52);
   --tx3:rgba(226,232,240,.34);
   --panel:rgba(0,0,0,.60);
   --glow:inset 0 0 20px rgba(226,232,240,.05);
   /* Two registers. Mono carries DATA — tags, counts, line numbers, code — and
      that is what makes the screen read as an instrument. Sans carries
      LANGUAGE. Space Mono is a display face: at 10px, tracked out and
      uppercased, a sentence set in it is genuinely hard to read, and the
      people using this are not going to squint at a rig. */
   --mono:'JetBrains Mono','Space Mono',ui-monospace,'SF Mono',SFMono-Regular,Menlo,monospace;
   --sans:'Inter','SF Pro Text',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
   /* four L-brackets, drawn as background layers so no extra DOM is needed */
   --bk:
     linear-gradient(var(--l4),var(--l4)) 0 0/11px 2px no-repeat,
     linear-gradient(var(--l4),var(--l4)) 0 0/2px 11px no-repeat,
     linear-gradient(var(--l4),var(--l4)) 100% 0/11px 2px no-repeat,
     linear-gradient(var(--l4),var(--l4)) 100% 0/2px 11px no-repeat,
     linear-gradient(var(--l4),var(--l4)) 0 100%/11px 2px no-repeat,
     linear-gradient(var(--l4),var(--l4)) 0 100%/2px 11px no-repeat,
     linear-gradient(var(--l4),var(--l4)) 100% 100%/11px 2px no-repeat,
     linear-gradient(var(--l4),var(--l4)) 100% 100%/2px 11px no-repeat;
   /* CRT texture. Kept light: at .22 it reads as damage on a data table
      rather than as atmosphere. */
   --scan:repeating-linear-gradient(180deg,
      rgba(0,0,0,0) 0 2px, rgba(0,0,0,.06) 2px 4px);
   /* indicator lamps — the only colour in the interface */
   --ok:#4ade80; --acc:#7dd3fc; --warn:#fbbf24; --bad:#f87171; --alt:#c4b5fd;
   --r:0px;
 }
 /* Streamlit's own markdown rules out-specify a bare class selector, so custom
    type was silently rendering at its 1rem default — the section labels were
    coming out at 16px instead of 9.5px, which is most of why this looked
    generic. Font and size are pinned. */
 html,body,.stApp,[class*="css"],input,textarea,button,select,
 [data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] *,
 [data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *,
 .stAlert,.stAlert *,[data-testid="stChatMessage"] p,label,label *,
 [data-baseweb="tab"],[data-baseweb="tab"] *,[data-testid="stExpander"] summary,
 [data-testid="stFileUploader"],[data-testid="stFileUploader"] *{
   font-family:var(--sans)!important;-webkit-font-smoothing:antialiased;}
 /* ...and mono wherever the content is a value rather than a sentence. */
 code,pre,kbd,samp,.mono,.display,.stat .v,.hud-hd,.eyebrow,.path,.chip,
 .telem,.telem *,.tl-c,.tl-seq,.pill,.pill b,.mast h1,.mast .sub,.mast .mark,
 table.tbl,table.tbl th,table.tbl td,[data-testid="stDataFrame"] *,
 .stat .k,.step-num,.data-pill,.stage .dt,.seg,
 [data-testid="stMarkdownContainer"] code,[data-testid="stCaptionContainer"] code{
   font-family:var(--mono)!important;}
 span[data-testid="stIconMaterial"],.material-icons,[class*="material-symbols"],
 [data-testid="stExpanderToggleIcon"]{font-family:'Material Symbols Rounded',
   'Material Icons'!important;}

 /* ── the stage ──────────────────────────────────────────── */
 .stApp{background:var(--bg);}
 .stApp::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
   background-image:
     linear-gradient(rgba(226,232,240,.045) 1px,transparent 1px),
     linear-gradient(90deg,rgba(226,232,240,.045) 1px,transparent 1px);
   background-size:50px 50px;}
 /* a vignette so the grid falls away at the edges instead of tiling forever */
 .stApp::after{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
   background:radial-gradient(ellipse 140% 110% at 50% 40%,
     transparent 32%, rgba(3,3,3,.72) 78%, rgba(3,3,3,.94) 100%);}
 /* Lift real content above the fixed grid/vignette layers WITHOUT changing
    its position: Streamlit keeps [stAppViewContainer] position:absolute,inset:0
    so the inner main can scroll. Forcing position:relative on it here made it
    grow to full content height and .stApp's overflow:hidden then clipped the
    bottom — the page could not be scrolled at all. z-index alone works because
    the container is already positioned. */
 [data-testid="stAppViewContainer"]{z-index:1;}
 section[data-testid="stSidebar"]{z-index:2;}
 .block-container{padding:1.5rem 2.2rem 4rem!important;max-width:100%!important;}
 /* Vertical rhythm. Streamlit stacks elements with a tight default gap, which
    is what makes a panel read as "everything jammed together". A little more
    air between blocks, and more between a widget and its label. */
 [data-testid="stMain"] [data-testid="stVerticalBlock"]{gap:1rem;}
 section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.7rem;}
 [data-testid="stElementContainer"]{margin:0;}
 [data-testid="stWidgetLabel"]{margin-bottom:5px;}
 [data-testid="stHorizontalBlock"]{gap:.9rem;}
 #MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
 h1,h2,h3,h4,h5{color:var(--tx);letter-spacing:.06em;font-weight:400;
   text-transform:uppercase;}
 h3{font-size:15px;letter-spacing:.16em;}

 /* the panel primitive: square, bracketed, scanlined */
 .hudp{position:relative;background:var(--bk),var(--panel);
   border:1px solid var(--l3);box-shadow:var(--glow);
   backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);}
 .hudp::after{content:'';position:absolute;inset:0;pointer-events:none;
   background:var(--scan);z-index:0;}

 /* a panel header — system name left, state right */
 .hud-hd{display:flex;justify-content:space-between;align-items:center;
   font-size:11px!important;letter-spacing:.14em!important;text-transform:uppercase;color:var(--tx);
   border-bottom:1px solid var(--l1);padding-bottom:8px;margin-bottom:15px;}
 .hud-hd .st{color:var(--tx3);letter-spacing:.16em;}

 /* segment meter */
 .seg{display:inline-flex;gap:1px;padding:1px;border:1px solid var(--l2);
   height:12px;vertical-align:middle;}
 .seg i{width:5px;background:rgba(226,232,240,.10);display:block;}
 .seg i.on{background:var(--fg);}

 /* ── masthead ───────────────────────────────────────────── */
 .mast{display:flex;align-items:center;gap:20px;flex-wrap:wrap;
   position:relative;background:var(--bk),var(--panel);
   border:1px solid var(--l3);box-shadow:var(--glow);
   padding:11px 20px;margin:0 0 18px;}
 .mast .logo{display:flex;align-items:center;gap:11px;flex:none;}
 .mast .mark{width:22px;height:22px;flex:none;border:1px solid var(--l3);
   display:grid;place-items:center;color:var(--tx);font-weight:700;font-size:11px;}
 .mast h1{font-size:14px!important;margin:0;font-weight:700;letter-spacing:.16em;
   white-space:nowrap;line-height:1.2;color:var(--tx);text-transform:uppercase;}
 .mast .sub{font-size:10px!important;color:var(--tx3);letter-spacing:.12em;line-height:1.5;
   text-transform:uppercase;font-weight:400;margin-top:3px;white-space:nowrap;}
 .mast .pills{display:flex;gap:0;flex-wrap:wrap;justify-content:flex-end;
   margin-left:auto;align-items:center;row-gap:6px;}
 @media(max-width:1100px){
   .mast{flex-direction:column;align-items:stretch;gap:10px;}
   .mast .pills{justify-content:flex-start;margin-left:0;
     border-top:1px solid var(--l1);padding-top:9px;}
   .mast .pill:first-child{padding-left:0;}
 }
 @media(max-width:1280px){ .mast .pill.opt{display:none;} }
 @media(max-width:900px){ .mast .sub{display:none;} }

 .pill{display:inline-flex;align-items:center;gap:7px;background:transparent;
   border:0;padding:0 15px;font-size:11px!important;color:var(--tx2);white-space:nowrap;
   font-weight:400;letter-spacing:.1em;text-transform:uppercase;}
 .pill + .pill{border-left:1px solid var(--l1);}
 .pill b{color:var(--tx);font-weight:700;font-size:12.5px;letter-spacing:.02em;}
 .pill .d{width:6px;height:6px;border-radius:50%;background:var(--ok);flex:none;
   box-shadow:0 0 7px var(--ok);animation:blink 2.4s linear infinite;}
 @keyframes blink{48%{opacity:1}50%{opacity:.25}52%{opacity:1}}

 /* ── sidebar ──────────────────────────────────────────────
    A bracketed panel box was wrapping the whole sidebar. Its border is drawn
    once at the box's height, but the expanders inside grow and shrink, so the
    frame ended up cutting straight through the content. The sidebar is a
    scrolling column of controls, not a fixed panel — so it gets a clean edge
    and a solid ground, and each section breathes on its own. */
 section[data-testid="stSidebar"]{background:#050505!important;
   border-right:1px solid var(--l2)!important;width:312px!important;}
 section[data-testid="stSidebar"] > div,
 section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
   background:transparent!important;}
 section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{
   padding:8px 12px 0 18px!important;}
 section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{
   padding:14px 18px 28px!important;}

 /* a section label doubles as a panel header */
 .eyebrow{font-size:11px!important;letter-spacing:.13em!important;text-transform:uppercase;
   color:var(--tx);margin:0 0 12px;font-weight:500;line-height:1.6;
   display:flex;align-items:center;gap:10px;
   border-bottom:1px solid var(--l1);padding-bottom:6px;}
 .eyebrow::after{content:'';flex-grow:1;height:1px;background:transparent;}
 .eyebrow.sp{margin-top:24px;}
 .eyebrow.plain{border-bottom:0;padding-bottom:0;}

 /* ── surfaces ───────────────────────────────────────────── */
 .card{position:relative;background:var(--panel);border:1px solid var(--l2);
   padding:14px 16px;color:var(--txb);}
 .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px;}
 .stat{position:relative;background:var(--bk),var(--panel);
   border:1px solid var(--l3);box-shadow:var(--glow);padding:14px 16px 15px;
   overflow:hidden;}
 .stat::after{content:'';position:absolute;inset:0;pointer-events:none;
   background:var(--scan);}
 .stat .k{font-size:10.5px!important;letter-spacing:.13em!important;text-transform:uppercase;
   color:var(--tx2);margin-bottom:9px;font-weight:500;
   border-bottom:1px solid var(--l1);padding-bottom:5px;}
 .stat .v{font:700 30px/1 var(--mono)!important;color:var(--tx);
   letter-spacing:-.01em!important;}
 .stat .v.acc,.stat .v.ok,.stat .v.warn,.stat .v.bad{color:var(--tx);}
 .stat .v.ok::after,.stat .v.warn::after,.stat .v.bad::after,.stat .v.acc::after{
   content:'';display:inline-block;width:6px;height:6px;border-radius:50%;
   margin-left:10px;vertical-align:middle;}
 .stat .v.ok::after{background:var(--ok);box-shadow:0 0 7px var(--ok);}
 .stat .v.acc::after{background:var(--acc);box-shadow:0 0 7px var(--acc);}
 .stat .v.warn::after{background:var(--warn);box-shadow:0 0 7px var(--warn);}
 .stat .v.bad::after{background:var(--bad);box-shadow:0 0 7px var(--bad);}
 .stat .s{font-size:12px!important;color:var(--tx2);margin-top:10px;letter-spacing:0!important;
   line-height:1.5;}

 /* the display readout — big, mono, tracked out */
 .display{font-size:30px!important;font-weight:700;letter-spacing:.03em!important;line-height:1.18;
   color:var(--tx);margin:2px 0 7px;text-transform:uppercase;}
 .display-sub{font-size:14px!important;font-weight:400;color:var(--tx2);
   margin-bottom:18px;letter-spacing:.01em!important;line-height:1.5;}

 /* ── answer ─────────────────────────────────────────────── */
 .answer{position:relative;background:var(--bk),var(--panel);
   border:1px solid var(--l3);border-left:4px solid var(--l4);
   box-shadow:var(--glow);padding:17px 19px;margin:4px 0 12px;overflow:hidden;}
 .answer::after{content:'';position:absolute;inset:0;pointer-events:none;
   background:var(--scan);}
 .answer .txt{color:var(--tx);font-size:15.5px!important;line-height:1.72;font-weight:400;
   letter-spacing:0!important;position:relative;z-index:1;}
 .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px;position:relative;z-index:1;}
 .chip{border:1px solid var(--l2);color:var(--tx2);border-radius:0;
   padding:4px 10px;font-size:10.5px!important;background:rgba(226,232,240,.03);
   font-weight:400;letter-spacing:.1em;text-transform:uppercase;
   display:inline-flex;align-items:center;gap:7px;}
 .chip::before{content:'';width:5px;height:5px;border-radius:50%;
   background:var(--tx3);flex:none;}
 .chip.g::before{background:var(--ok);box-shadow:0 0 6px var(--ok);}
 .chip.d::before{background:var(--warn);box-shadow:0 0 6px var(--warn);}
 .path{margin-top:13px;padding-top:11px;border-top:1px dashed var(--l2);
   font:400 12px/1.9 var(--mono)!important;color:var(--tx2);word-break:break-word;
   letter-spacing:.01em;position:relative;z-index:1;}
 .path b{color:var(--tx);font-weight:700;}

 /* ── pipeline tracker ───────────────────────────────────── */
 .stage{display:flex;align-items:center;gap:12px;padding:8px 0;
   border-bottom:1px solid rgba(226,232,240,.06);}
 .stage .bul{width:14px;height:14px;border:1px solid var(--l2);flex:none;
   display:grid;place-items:center;font-size:8px;color:var(--bg);font-weight:700;}
 .stage.done .bul{background:var(--fg);border-color:var(--fg);}
 .stage.run .bul{border-color:var(--l4);box-shadow:0 0 0 3px rgba(226,232,240,.07);
   animation:blink 1.4s linear infinite;}
 .stage .nm{flex:1;font-size:13px!important;color:var(--tx2);letter-spacing:.01em;}
 .stage.done .nm{color:var(--tx);} .stage.run .nm{color:var(--tx);}
 .stage .dt{font:400 12px var(--mono);color:var(--tx3);letter-spacing:.02em;}

 /* ── rows and key-values ────────────────────────────────── */
 .rev{position:relative;background:var(--panel);border:1px solid var(--l2);
   padding:14px 16px;}
 .rd{font-size:14px!important;color:var(--txb);line-height:1.95;
   letter-spacing:0!important;}
 .rd b{color:var(--tx);font-weight:700;}
 .rd .bad{color:var(--bad);}
 .kv{font-size:14px!important;line-height:1.95;color:var(--txb);
   letter-spacing:0!important;}
 .kv b{color:var(--tx);font-weight:700;}
 .kv .m{color:var(--tx);font-size:13px;font-family:var(--mono)!important;}

 .step-block{padding:15px 17px;background:var(--panel);border:1px solid var(--l1);
   transition:.2s;}
 .step-block:hover{border-color:var(--l3);background:rgba(226,232,240,.04);}
 .step-num{font-size:10.5px!important;color:var(--tx3);margin-bottom:8px;letter-spacing:.16em;}
 .step-title{font-size:14px!important;font-weight:600;color:var(--tx);margin-bottom:7px;
   letter-spacing:.04em;}
 .step-desc{font-size:13.5px!important;color:var(--txb);line-height:1.62;letter-spacing:0;}
 .data-pill{display:flex;align-items:center;justify-content:space-between;
   padding:10px 13px;background:rgba(226,232,240,.035);font-size:13px!important;
   color:var(--tx);border-left:3px solid var(--l3);margin-bottom:7px;
   letter-spacing:.01em;}
 .data-pill .metric{font-size:12px;color:var(--tx2);letter-spacing:.04em;}

 /* ── widgets ────────────────────────────────────────────── */
 .stTextInput input,.stTextArea textarea{background:rgba(0,0,0,.55)!important;
   border:1px solid var(--l2)!important;color:var(--tx)!important;
   border-radius:0!important;font-size:15px!important;padding:13px 15px!important;
   font-family:var(--sans)!important;font-weight:400!important;
   letter-spacing:0!important;transition:.15s;}
 .stTextInput input::placeholder,.stTextArea textarea::placeholder{
   color:var(--tx3)!important;letter-spacing:0!important;font-size:14px!important;}
 .stTextInput input:focus,.stTextArea textarea:focus{
   border-color:var(--l4)!important;background:rgba(226,232,240,.05)!important;
   box-shadow:var(--glow)!important;}
 .stSelectbox div[data-baseweb="select"]>div{background:rgba(0,0,0,.55)!important;
   border-color:var(--l2)!important;border-radius:0!important;color:var(--tx)!important;
   font-size:13.5px!important;letter-spacing:0!important;}
 div[data-baseweb="popover"] ul{background:#050505!important;
   border:1px solid var(--l3)!important;border-radius:0!important;}
 div[data-baseweb="popover"] li{font-size:13.5px!important;color:var(--txb)!important;
   letter-spacing:0!important;font-family:var(--sans)!important;padding:9px 12px!important;}

 .stButton>button{border-radius:0;border:1px solid var(--l2);
   background:rgba(226,232,240,.03);font-size:13px;font-weight:500;transition:.15s;
   padding:11px 15px;min-width:0;min-height:40px;line-height:1.4;letter-spacing:.01em;
   text-transform:none;white-space:normal;}
 .stButton>button *{color:var(--txb)!important;font-weight:500!important;
   letter-spacing:.01em!important;font-size:13px!important;text-transform:none;}
 .stButton>button:hover{border-color:var(--l4);background:rgba(226,232,240,.08);}
 .stButton>button:hover *{color:var(--tx)!important;}
 .stButton>button[kind="primary"]{background:var(--fg);border-color:var(--fg);}
 .stButton>button[kind="primary"] *{color:#030303!important;font-weight:700!important;}
 .stButton>button[kind="primary"]:hover{background:#fff;border-color:#fff;}
 /* form submit buttons use a different testid; give them the same look */
 [data-testid="stFormSubmitButton"]>button{border-radius:0;border:1px solid var(--l2);
   background:rgba(226,232,240,.03);font-size:13px;font-weight:500;padding:11px 15px;
   min-height:40px;letter-spacing:.01em;text-transform:none;transition:.15s;}
 [data-testid="stFormSubmitButton"]>button *{color:var(--txb)!important;
   font-weight:500!important;font-size:13px!important;text-transform:none;}
 [data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"]{
   background:var(--fg)!important;border-color:var(--fg)!important;}
 [data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"] *{
   color:#030303!important;font-weight:700!important;}
 [data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"]:hover{
   background:#fff!important;border-color:#fff!important;}
 [data-testid="stForm"]{border:0!important;padding:0!important;}
 .stDownloadButton>button{border-radius:0;border:1px solid var(--l1);
   background:transparent;font-size:12.5px;width:100%;padding:10px 13px;
   letter-spacing:.01em;text-transform:none;}
 .stDownloadButton>button *{color:var(--tx2)!important;letter-spacing:.01em!important;
   font-size:12.5px!important;text-transform:none;}
 .stDownloadButton>button:hover{border-color:var(--l3);}
 .stDownloadButton>button:hover *{color:var(--tx)!important;}
 /* suggestions are sentences, not controls: they wrap */
 .sugg .stButton>button{background:transparent;border-color:var(--l1);
   text-align:left;justify-content:flex-start;font-size:13px;padding:12px 15px;
   text-transform:none;letter-spacing:0;white-space:normal;line-height:1.5;}
 .sugg .stButton>button *{letter-spacing:0!important;
   font-size:13px!important;text-transform:none;text-align:left;}

 .stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--l2);
   margin-bottom:18px;}
 .stTabs [data-baseweb="tab"]{background:transparent;color:var(--tx2);font-size:12.5px;
   font-weight:500;padding:13px 20px;letter-spacing:.08em;text-transform:uppercase;
   border-right:1px solid rgba(226,232,240,.07);}
 .stTabs [data-baseweb="tab"]:hover{color:var(--tx2);background:rgba(226,232,240,.03);}
 .stTabs [aria-selected="true"]{color:var(--tx)!important;
   background:rgba(226,232,240,.06)!important;}
 .stTabs [data-baseweb="tab-highlight"]{background:var(--fg)!important;height:2px!important;}
 .stTabs [data-baseweb="tab-border"]{display:none!important;}

 details,.stExpander{border:1px solid var(--l2)!important;border-radius:0!important;
   background:rgba(255,255,255,.015)!important;margin-bottom:10px;}
 .stExpander summary,.stExpander details>summary{font-size:13px!important;color:var(--txb)!important;
   letter-spacing:.01em!important;text-transform:none;padding:11px 14px!important;}
 .stExpander summary:hover{color:var(--tx)!important;background:rgba(226,232,240,.03);}
 [data-testid="stExpander"] [data-testid="stExpanderDetails"]{padding:4px 14px 14px!important;}

 /* File uploader — Streamlit's default is a wide grey slab with a pill button;
    that is the "looks like a child drew it" box. Rebuilt as a quiet drop zone:
    a single dashed frame, centred prompt, and the Browse control styled like
    every other button in the app rather than a baseweb default. */
 [data-testid="stFileUploader"]{background:transparent;border:0;padding:0;}
 [data-testid="stFileUploader"] > section,
 [data-testid="stFileUploaderDropzone"]{
   flex-direction:column!important;align-items:center!important;text-align:center;
   gap:9px!important;padding:20px 16px!important;background:rgba(226,232,240,.02)!important;
   border:1px dashed var(--l2)!important;border-radius:0!important;transition:.15s;}
 [data-testid="stFileUploaderDropzone"]:hover{border-color:var(--l3)!important;
   background:rgba(226,232,240,.045)!important;}
 [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"]{
   display:flex;flex-direction:column;align-items:center;gap:2px;color:var(--tx2)!important;}
 [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span{
   font-size:13px!important;color:var(--txb)!important;font-weight:500;letter-spacing:0!important;}
 [data-testid="stFileUploader"] small{color:var(--tx3)!important;font-size:11px!important;
   letter-spacing:0!important;}
 /* the Browse button, matched to the app's control style */
 [data-testid="stFileUploader"] button,
 [data-testid="stBaseButton-secondary"]{
   border-radius:0!important;border:1px solid var(--l2)!important;
   background:rgba(226,232,240,.04)!important;color:var(--txb)!important;
   font-size:12.5px!important;font-weight:500!important;letter-spacing:.01em!important;
   text-transform:none!important;padding:8px 16px!important;box-shadow:none!important;}
 [data-testid="stFileUploader"] button:hover{border-color:var(--l4)!important;
   background:rgba(226,232,240,.09)!important;color:var(--tx)!important;}
 /* uploaded-file chip */
 [data-testid="stFileUploaderFile"]{background:rgba(226,232,240,.04)!important;
   border:1px solid var(--l1)!important;border-radius:0!important;padding:8px 10px!important;
   margin-top:8px;}
 [data-testid="stFileUploaderFileName"]{font-size:12.5px!important;color:var(--tx)!important;}
 .stCheckbox label span{font-size:13.5px!important;color:var(--txb)!important;
   letter-spacing:0!important;}
 [data-testid="stCheckbox"] span[data-baseweb="checkbox"] div{border-radius:0!important;}
 .stSlider label,.stRadio label span{font-size:13.5px!important;color:var(--txb)!important;
   letter-spacing:0!important;}
 code{color:var(--tx)!important;background:rgba(226,232,240,.07)!important;
   font-family:var(--mono)!important;font-size:12px!important;
   border-radius:0!important;padding:2px 6px!important;}
 .stCodeBlock,pre{border-radius:0!important;border:1px solid var(--l1)!important;
   background:rgba(0,0,0,.55)!important;}
 hr{border-color:var(--l1);margin:16px 0;}
 .stAlert{border-radius:0!important;font-size:13.5px!important;
   border:1px solid var(--l2)!important;background:var(--panel)!important;
   color:var(--txb)!important;}
 .stAlert p{color:var(--txb)!important;letter-spacing:0!important;
   font-size:13.5px!important;line-height:1.6!important;}
 [data-testid="stCaptionContainer"]{color:var(--tx2)!important;font-size:12.5px!important;
   letter-spacing:0!important;line-height:1.6!important;}
 [data-testid="stCaptionContainer"] *{font-size:12.5px!important;letter-spacing:0!important;}
 [data-testid="stChatMessage"]{background:var(--panel)!important;
   border:1px solid var(--l1)!important;border-radius:0!important;}
 [data-testid="stChatMessageAvatarUser"],[data-testid="stChatMessageAvatarAssistant"]{
   background:transparent!important;border:1px solid var(--l2)!important;
   border-radius:0!important;color:var(--tx2)!important;}
 [data-testid="stDataFrame"]{border:1px solid var(--l2)!important;
   border-radius:0!important;overflow:hidden;background:rgba(0,0,0,.5);}
 [data-testid="stDataFrame"] *{font-family:var(--mono)!important;}

 ::-webkit-scrollbar{width:5px;height:5px;}
 ::-webkit-scrollbar-track{background:#000;border-left:1px solid var(--l1);}
 ::-webkit-scrollbar-thumb{background:rgba(226,232,240,.42);border-radius:0;}
 ::-webkit-scrollbar-thumb:hover{background:var(--fg);}

 /* ── panel primitive for st.container(border=True) ──────────
    Streamlit wraps EVERY vertical block in the same testid, bordered or not,
    so styling that testid alone framed each tab panel a second time. A panel
    opts in: `theme.PANEL` drops an invisible marker as the container's first
    child and `:has()` matches only the wrapper that owns it. */
 .pnl-mark{display:none;}
 [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div:first-child .pnl-mark){
   position:relative;background:var(--bk),var(--panel);
   border:1px solid var(--l3)!important;border-radius:0!important;
   box-shadow:var(--glow);padding:18px 20px!important;}
 [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div:first-child .pnl-mark)::after{
   content:'';position:absolute;inset:0;pointer-events:none;
   background:var(--scan);z-index:0;}
 /* an image inside a panel is content, not chrome — it sits above the wash */
 [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stImage"],
 [data-testid="stVerticalBlockBorderWrapper"] iframe{position:relative;z-index:1;}
 [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div:first-child .pnl-mark)
   [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div:first-child .pnl-mark){
   background:var(--panel);border-color:var(--l1)!important;box-shadow:none;
   padding:14px 16px!important;}
 [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div:first-child .pnl-mark)
   [data-testid="stVerticalBlockBorderWrapper"]:has(> div > div:first-child .pnl-mark)::after{
   display:none;}

 /* Streamlit columns do not wrap, so a four-across row overlapped itself on a
    narrow window. */
 @media(max-width:1200px){
   [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;}
   [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
     min-width:46%!important;flex:1 1 46%!important;}
 }
 @media(max-width:760px){
   [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
     min-width:100%!important;flex:1 1 100%!important;}
   .block-container{padding:1rem 1rem 3rem!important;}
 }

 /* ── telemetry strip ────────────────────────────────────── */
 .telem{position:fixed;left:0;right:0;bottom:0;z-index:60;height:26px;
   background:rgba(0,0,0,.92);border-top:1px solid var(--l3);
   box-shadow:0 -8px 24px rgba(0,0,0,.6);}
 .tl-in{display:flex;align-items:center;height:100%;gap:0;padding:0 16px;
   overflow:hidden;white-space:nowrap;}
 .tl-c{display:inline-flex;align-items:center;gap:8px;padding:0 16px;
   font-size:10px!important;letter-spacing:.12em;text-transform:uppercase;
   color:var(--tx);border-right:1px solid var(--l1);height:100%;}
 .tl-c:first-child{padding-left:0;}
 .tl-c i{font-style:normal;color:var(--tx3);}
 .tl-seq{margin-left:auto;font-size:10px!important;letter-spacing:.12em;
   color:var(--tx3);text-transform:uppercase;padding-left:16px;}
 /* keep the last row of every screen clear of the strip */
 .block-container{padding-bottom:4.5rem!important;}

 /* ── results table ──────────────────────────────────────── */
 .tbl-wrap{max-height:440px;overflow:auto;border:1px solid var(--l2);
   background:rgba(0,0,0,.5);}
 table.tbl{width:100%;border-collapse:collapse;font-size:12.5px;}
 table.tbl th{position:sticky;top:0;background:#070707;color:var(--tx3);
   text-align:left;font-weight:500;font-size:10.5px;letter-spacing:.12em;
   text-transform:uppercase;padding:11px 13px;border-bottom:1px solid var(--l2);}
 table.tbl td{padding:10px 13px;border-bottom:1px solid rgba(226,232,240,.06);
   color:var(--txb);vertical-align:top;letter-spacing:0;line-height:1.55;}
 table.tbl tr:hover td{background:rgba(226,232,240,.05);}
 table.tbl td.m{color:var(--tx);font-size:12px;}
 table.tbl td.ok,table.tbl td.bad{color:var(--tx2);text-transform:uppercase;
   letter-spacing:.08em;font-size:11px;}
 table.tbl td.ok::before,table.tbl td.bad::before{content:'';display:inline-block;
   width:6px;height:6px;border-radius:50%;margin-right:8px;vertical-align:middle;}
 table.tbl td.ok::before{background:var(--ok);box-shadow:0 0 6px var(--ok);}
 table.tbl td.bad::before{background:var(--bad);box-shadow:0 0 6px var(--bad);}

 /* the one paragraph on the front page that has to land */
 .lead{font-size:16.5px!important;line-height:1.68;color:var(--txb);
   letter-spacing:0!important;margin:2px 0 16px;max-width:70ch;}
 .lead b{color:var(--tx);font-weight:600;}

 .empty{position:relative;border:1px dashed var(--l2);padding:44px 26px;
   text-align:center;color:var(--tx2);font-size:14px!important;line-height:1.7;
   background:rgba(0,0,0,.4);letter-spacing:0;}
 .empty b{display:block;color:var(--tx);font-size:19px!important;margin-bottom:12px;
   font-weight:600;letter-spacing:.06em;}
</style>
"""
