#!/usr/bin/env python3
"""
Daily US News Brief — Cloud Pipeline v2
Fetches news via NewsAPI → generates TOEIC 600+ HTML via DeepSeek → outputs index.html + archive
"""
import os, json, re, sys
from datetime import datetime, timezone, timedelta

NEWSAPI_KEY  = os.environ["NEWSAPI_KEY"]
DEEPSEEK_KEY = os.environ["DEEPSEEK_KEY"]
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
BEIJING_TZ   = timezone(timedelta(hours=8))
ARCHIVE_DIR  = "archive"
TODAY        = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")

CATEGORIES = {"POLITICS": "politics", "ECONOMY": "business", "RESEARCH": "science"}

CREDIBLE_DOMAINS = {
    "reuters.com","apnews.com","nytimes.com","wsj.com","washingtonpost.com",
    "bloomberg.com","cnbc.com","bbc.com","bbc.co.uk","npr.org","politico.com",
    "theguardian.com","usatoday.com","axios.com","cbsnews.com","abcnews.go.com",
    "nbcnews.com","cnn.com","time.com","theatlantic.com","economist.com",
    "ft.com","science.org","nature.com","sciencedaily.com","newscientist.com",
    "space.com","phys.org","sciencenews.org","livescience.com",
    "nih.gov","nasa.gov","energy.gov","noaa.gov","nsf.gov",
}
SUSPECT_DOMAINS = {"medium.com","blogspot.com","wordpress.com","substack.com","yahoo.com","aol.com","msn.com"}

# ── NewsAPI ─────────────────────────────────────────────
def fetch_newsapi(category):
    import urllib.request as ur
    url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&pageSize=15&apiKey={NEWSAPI_KEY}"
    try:
        with ur.urlopen(ur.Request(url, headers={"User-Agent":"DailyENNews/2.0"}), timeout=15) as r:
            d = json.loads(r.read())
        return d.get("articles",[]) if d.get("status")=="ok" else []
    except Exception as e:
        print(f"  [ERROR] NewsAPI {category}: {e}"); return []

def source_quality(a):
    import urllib.parse as up
    sn = (a.get("source",{}) or {}).get("name","Unknown")
    dom = up.urlparse(a.get("url","")).netloc.lower().replace("www.","")
    flags, score = [], 5
    if any(c in dom for c in CREDIBLE_DOMAINS): score+=3; flags.append("trusted")
    elif any(s in dom for s in SUSPECT_DOMAINS): score-=2; flags.append("suspect")
    elif dom: score-=1; flags.append("unknown")
    if sn=="Unknown" or not sn: score-=2; flags.append("no-source")
    if len(a.get("content") or a.get("description") or "")<100: score-=2; flags.append("thin")
    if not a.get("publishedAt"): score-=1; flags.append("no-date")
    return {"score":max(0,score),"flags":flags,"domain":dom or sn,"source_name":sn}

def filter_articles(arts, min_score=4):
    for a in arts: a["_q"]=source_quality(a)
    arts.sort(key=lambda a:a["_q"]["score"], reverse=True)
    return [a for a in arts if a["_q"]["score"]>=min_score][:6]

def build_summary(cat, arts):
    lines=[f"=== {cat} ({len(arts)} articles) ==="]
    for i,a in enumerate(arts[:6],1):
        q=a["_q"]
        lines.append(f"[{i}] {a.get('title','?')}\n    {q['source_name']} | {a.get('publishedAt','')[:10]} | Q{q['score']}/8\n    {a.get('description') or a.get('content','')}")
    return "\n".join(lines)

# ── DeepSeek ────────────────────────────────────────────
SYSTEM_PROMPT_ARTICLES = """You are an expert English teacher and news editor creating a daily reading page for a Chinese student at TOEIC ~600 (CEFR B1).

## OUTPUT — valid JSON only (NO dictionary field):
{
  "date": "Thursday, July 30, 2026",
  "politics": [{"title":"...", "paragraphs":["...","...","..."], "translation":["中文...","中文...","中文..."], "source":"Source: X — YYYY-MM-DD"}, ...],
  "economy": [{"title":"...", "paragraphs":["...","...","..."], "translation":["中文...","中文...","中文..."], "what_it_means":"...", "source":"Source: X — YYYY-MM-DD"}, ...],
  "research": [{"title":"...", "paragraphs":["...","...","..."], "translation":["中文...","中文...","中文..."], "source":"Source: X — YYYY-MM-DD"}, ...]
}

## REQUIREMENTS
- 2-3 articles per category. Each: exactly 3 English paragraphs (2-4 sentences) + 3 Chinese translations.
- English B1+ to B2. Explain technical terms in parentheses.
- Economy articles MUST include "what_it_means" field.
- All sources dated within last 2 days. No single person dominates. No C1+ rare words."""

SYSTEM_PROMPT_DICT = """You are a TOEIC vocabulary expert. Given English news articles, build a COMPLETE English-Chinese dictionary.

## OUTPUT — valid JSON only:
{"dictionary": {"word":"pos. 中文释义", ...}}

## STRICT REQUIREMENTS
- MUST contain AT LEAST 500 entries.
- Cover EVERY non-trivial word (nouns, verbs, adjectives, adverbs) from the articles.
- Include multi-word phrases like "credit default swap" or "ground game".
- Format: ALL keys lowercase. Value format: "pos. 中文释义".
- Go through each article paragraph word by word. Do not skip common words — a B1 learner may not know them."""

DICT_RETRY_PROMPT = "\n\n## ATTENTION\nThe dictionary only contained {count} entries — less than 500. REGENERATE with at least 500 entries covering every word from the articles."

def call_deepseek(system: str, user_msg: str, max_tok: int = 8000, use_reasoner: bool = False) -> dict:
    import urllib.request as ur
    model = "deepseek-reasoner" if use_reasoner else "deepseek-chat"
    payload = {
        "model": model,
        "messages":[{"role":"system","content":system},{"role":"user","content":user_msg}],
        "max_tokens": max_tok,
    }
    if not use_reasoner:
        payload["temperature"] = 0.7
    if use_reasoner or model == "deepseek-chat":
        payload["response_format"] = {"type":"json_object"}
    req = ur.Request(DEEPSEEK_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {DEEPSEEK_KEY}"})
    try:
        with ur.urlopen(req, timeout=180) as r: result = json.loads(r.read())
    except Exception as e:
        print(f"[ERROR] DeepSeek: {e}"); sys.exit(1)
    content = result["choices"][0]["message"]["content"]
    try: return json.loads(content)
    except json.JSONDecodeError: return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$","",content.strip()))

def generate_content(news_summary: str) -> dict:
    """Two-step: articles (chat) + dictionary (reasoner, 64K max output)."""
    today = datetime.now(BEIJING_TZ).strftime('%A, %B %d, %Y')
    
    # Step 1: Articles + translations (fast, cheap)
    print(f"\n[STEP 1/2] Generating articles (deepseek-chat)...")
    data = call_deepseek(SYSTEM_PROMPT_ARTICLES, f"Today is {today}.\n\n{news_summary}", max_tok=6000)
    n_arts = sum(len(data.get(k,[])) for k in ['politics','economy','research'])
    print(f"  Articles: {n_arts}")

    # Step 2: Dictionary (deepseek-reasoner, 64K max output, single call)
    print(f"[STEP 2/2] Generating dictionary (deepseek-reasoner, max_tok=30000)...")
    arts_json = json.dumps({k: data.get(k,[]) for k in ['politics','economy','research']}, ensure_ascii=False)
    d = call_deepseek(SYSTEM_PROMPT_DICT, f"Generate dictionary for these articles:\n{arts_json}", max_tok=30000, use_reasoner=True)
    dcount = len(d.get("dictionary", {}))
    print(f"  Dict entries: {dcount}")
    
    if dcount < 500:
        print(f"  [RETRY] Dict too short ({dcount}<500)")
        try:
            d2 = call_deepseek(SYSTEM_PROMPT_DICT, arts_json + "\n" + DICT_RETRY_PROMPT.replace("{count}", str(dcount)), max_tok=30000, use_reasoner=True)
            all_dict = {}
            all_dict.update(d.get("dictionary", {}))
            all_dict.update(d2.get("dictionary", {}))
            data["dictionary"] = all_dict
            print(f"  Dict after retry: {len(all_dict)} entries (merged)")
        except Exception:
            data["dictionary"] = d.get("dictionary", {})
            print(f"  Retry failed, using {dcount} entries")
    else:
        data["dictionary"] = d.get("dictionary", {})
    return data

# ── HTML Template ───────────────────────────────────────
def build_html(data: dict) -> str:
    date_str = data["date"]
    dict_entries = ",\n".join(f'    "{k}":"{v}"' for k,v in data.get("dictionary",{}).items())
    sections, total_articles = "", 0

    for label, accent_cls, emoji in [
        ("politics","politics","&#127987;"),
        ("economy","economy","&#128176;"),
        ("research","research","&#128300;")
    ]:
        arts = data.get(label, [])
        if not arts: continue
        sections += f'<div class="section-title {accent_cls}">{emoji} {label.upper()}</div>\n'
        for art in arts:
            total_articles += 1
            paras_en = "".join(f"<p>{p}</p>\n" for p in art.get("paragraphs",[]))
            trans = art.get("translation",[])
            trans_html = ""
            if trans:
                trans_items = "".join(f"<p>{t}</p>\n" for t in trans)
                trans_html = f"""<details class="trans-toggle">
    <summary>&#127760; Chinese Translation (点击展开)</summary>
    <div class="trans-content">{trans_items}</div>
</details>"""
            wim = art.get("what_it_means","")
            wim_html = f'<p class="wim">&#128161; <strong>What this means for you:</strong> {wim}</p>\n' if wim else ""
            sections += f"""<div class="news-card">
    <h3>{total_articles}. {art["title"]}</h3>
    {paras_en}{wim_html}    {trans_html}
    <p class="source">{art["source"]}</p>
  </div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Today's US News — {date_str} | English Reading Practice</title>
<style>
  :root{{
    --bg:#faf9f7;--card-bg:#fff;--text:#2d2d2d;--text-light:#5a5a5a;--accent-p:#b22234;--accent-e:#1a5f8a;--accent-r:#2d7d46;
    --gold:#b7933a;--border:#e8e4df;--tag-bg:#f0ede8;--shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
    --shadow-lg:0 12px 40px rgba(0,0,0,.12);--radius:10px;--sidebar-w:200px;
  }}
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:Georgia,'Times New Roman',serif;background:var(--bg);color:var(--text);line-height:1.8;display:flex;min-height:100vh}}
  #sidebar{{
    position:fixed;left:0;top:0;width:var(--sidebar-w);height:100vh;background:#2c2c2c;color:#ccc;
    overflow-y:auto;padding:20px 0;z-index:100;font-family:-apple-system,sans-serif;font-size:.82rem;
    transition:transform .25s ease
  }}
  #sidebar h3{{color:#fff;font-size:.85rem;padding:8px 16px;border-bottom:1px solid #444;margin-bottom:8px;text-align:center}}
  #sidebar .date-link{{display:block;padding:6px 16px;color:#aaa;text-decoration:none;transition:all .15s;border-left:3px solid transparent}}
  #sidebar .date-link:hover{{color:#fff;background:#3a3a3a;border-left-color:var(--gold)}}
  #sidebar .date-link.active{{color:#fff;background:#3a3a3a;border-left-color:var(--gold);font-weight:600}}
  #sidebar .no-archive{{padding:16px;color:#666;font-size:.78rem;text-align:center}}
  #main{{margin-left:var(--sidebar-w);flex:1;max-width:780px;padding:32px 28px 60px}}
  header{{text-align:center;padding:36px 0 28px;border-bottom:2px solid var(--border);margin-bottom:32px}}
  header h1{{font-size:1.75rem;font-weight:700;color:#1a1a1a}}
  header .date{{font-size:.9rem;color:var(--text-light)}}
  .meta-bar{{display:flex;justify-content:center;gap:24px;flex-wrap:wrap;margin-top:14px;font-size:.82rem;color:var(--text-light)}}
  .meta-bar span{{background:var(--tag-bg);padding:4px 14px;border-radius:20px}}
  .section{{margin-bottom:40px}}
  .section-title{{font-size:1.25rem;font-weight:700;margin-bottom:16px;padding-bottom:6px;border-bottom:3px solid;display:flex;align-items:center;gap:8px}}
  .section-title.politics{{color:var(--accent-p);border-color:var(--accent-p)}}
  .section-title.economy{{color:var(--accent-e);border-color:var(--accent-e)}}
  .section-title.research{{color:var(--accent-r);border-color:var(--accent-r)}}
  .news-card{{background:var(--card-bg);border:1px solid var(--border);border-radius:var(--radius);padding:22px 26px;margin-bottom:16px;box-shadow:var(--shadow);transition:box-shadow .3s,border-color .3s}}
  .news-card:hover{{box-shadow:0 3px 12px rgba(0,0,0,.08);border-color:#d8d2c8}}
  .news-card h3{{font-size:1.08rem;font-weight:700;margin-bottom:10px;line-height:1.45}}
  .news-card p{{font-size:.95rem;color:var(--text-light);margin-bottom:12px;text-align:justify}}
  .news-card .source{{font-size:.78rem;color:#999;font-style:italic}}
  .news-card .wim{{color:#8a6d20;font-size:.88rem;padding:6px 0 0}}
  .trans-toggle{{margin:8px 0 12px;border:1px solid #e0d8c0;border-radius:8px;overflow:hidden}}
  .trans-toggle summary{{padding:8px 14px;background:#faf6ec;cursor:pointer;font-size:.85rem;color:#8a6d20;font-weight:600;user-select:none}}
  .trans-toggle .trans-content{{padding:12px 16px;background:#fffbee;color:#5a4a2a;font-size:.88rem;line-height:1.8}}
  .trans-toggle .trans-content p{{color:#5a4a2a!important;margin-bottom:8px!important}}
  .cloud-badge{{display:inline-block;background:#e8f5e9;color:#2d7d46;font-size:.7rem;padding:2px 10px;border-radius:10px;margin-left:8px;vertical-align:middle}}
  .tips{{background:#f4f9ff;border-radius:var(--radius);padding:16px 22px;margin-bottom:32px;font-size:.88rem;color:#3a5a7c;line-height:1.7;display:flex;align-items:flex-start;gap:12px}}
  .tips .icon{{font-size:1.4rem;flex-shrink:0}}
  .lookup-word{{cursor:pointer;border-bottom:1.5px dotted transparent;transition:border-color .2s,background .2s;border-radius:2px;padding:0 1px}}
  .lookup-word:hover{{border-bottom-color:var(--gold);background:rgba(183,147,58,.08)}}
  .lookup-word.active{{background:rgba(183,147,58,.18);border-bottom-color:var(--gold)}}
  #word-popover{{position:fixed;z-index:9999;background:#fffef9;border:1px solid #d8d0b8;border-radius:12px;box-shadow:var(--shadow-lg);padding:18px 22px;min-width:200px;max-width:320px;opacity:0;transform:translateY(8px) scale(.96);transition:opacity .2s cubic-bezier(.16,1,.3,1),transform .25s cubic-bezier(.16,1,.3,1);pointer-events:none;font-family:-apple-system,sans-serif;word-break:break-word}}
  #word-popover.visible{{opacity:1;transform:translateY(0) scale(1);pointer-events:auto}}
  #word-popover .pop-word{{font-size:1.25rem;font-weight:700;color:#1a1a1a;margin-bottom:4px;font-family:Georgia,serif}}
  #word-popover .pop-pos{{font-size:.75rem;color:#999;text-transform:uppercase;letter-spacing:.6px;margin-bottom:10px}}
  #word-popover .pop-meaning{{font-size:.92rem;color:#444;line-height:1.5;padding:8px 12px;background:#faf6ec;border-radius:6px;border-left:3px solid var(--gold)}}
  #word-popover .pop-meaning .zh{{color:#6b4c1e;font-weight:600}}
  #word-popover .pop-close{{position:absolute;top:8px;right:10px;width:24px;height:24px;border:none;background:none;cursor:pointer;font-size:1rem;color:#bbb;border-radius:50%;display:flex;align-items:center;justify-content:center;transition:color .2s,background .2s}}
  #word-popover .pop-close:hover{{color:#666;background:#f0ede8}}
  #sel-toolbar{{position:fixed;z-index:9998;background:#2d2d2d;color:#fff;border-radius:8px;padding:6px 14px;font-size:.82rem;font-family:-apple-system,sans-serif;cursor:pointer;opacity:0;transform:translateY(6px);transition:opacity .18s,transform .2s;pointer-events:none;box-shadow:0 6px 20px rgba(0,0,0,.25);white-space:nowrap}}
  #sel-toolbar.visible{{opacity:1;transform:translateY(0);pointer-events:auto}}
  #sel-toolbar:hover{{background:#3d3d3d}}
  #history-panel{{position:fixed;right:24px;bottom:80px;z-index:9990;display:flex;flex-direction:column;gap:6px;align-items:flex-end}}
  #history-toggle{{width:44px;height:44px;border-radius:50%;background:#fff;border:1px solid var(--border);box-shadow:0 2px 12px rgba(0,0,0,.08);cursor:pointer;font-size:1.2rem;display:flex;align-items:center;justify-content:center;transition:all .2s;position:relative}}
  #history-toggle:hover{{border-color:var(--gold);box-shadow:0 4px 16px rgba(0,0,0,.12)}}
  #history-badge{{position:absolute;top:-4px;right:-4px;background:var(--gold);color:#fff;font-size:.65rem;font-family:-apple-system,sans-serif;min-width:16px;height:16px;border-radius:8px;display:flex;align-items:center;justify-content:center;padding:0 4px;opacity:0;transform:scale(0);transition:all .3s cubic-bezier(.16,1,.3,1)}}
  #history-badge.show{{opacity:1;transform:scale(1)}}
  #history-list{{background:#fffef9;border:1px solid #d8d0b8;border-radius:12px;box-shadow:var(--shadow-lg);padding:14px 16px;max-height:300px;overflow-y:auto;width:240px;font-family:-apple-system,sans-serif;font-size:.82rem;display:none;flex-direction:column;gap:8px}}
  #history-list.open{{display:flex}}
  #history-list .h-item{{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;border-radius:6px;cursor:pointer;transition:background .15s;gap:8px}}
  #history-list .h-item:hover{{background:#faf6ec}}
  #history-list .h-word{{font-weight:600;color:#1a1a1a}}
  #history-list .h-zh{{color:#888;font-size:.78rem}}
  #history-clear{{text-align:center;font-size:.72rem;color:#bbb;cursor:pointer;padding:4px;border-radius:4px;transition:color .2s}}
  #history-clear:hover{{color:#c0392b}}
  footer{{text-align:center;padding:24px 0 0;border-top:1px solid var(--border);font-size:.78rem;color:#aaa;margin-top:8px}}
  .news-card p,.news-card h3{{-webkit-tap-highlight-color:transparent}}
  @media(max-width:760px){{
    #sidebar{{display:none}}
    #main{{margin-left:0;padding:20px 14px 40px}}
    .news-card{{padding:16px 18px}}
    #history-panel{{right:12px;bottom:60px}}
    #word-popover{{max-width:260px}}
  }}
  @media(pointer:coarse){{.news-card p,.news-card h3{{cursor:default}}}}
</style>
</head>
<body>
<div id="sidebar">
  <h3>&#128197; Past Editions</h3>
  <div id="sidebar-dates" class="no-archive">Loading...</div>
</div>
<div id="main">
<header>
  <h1>Today's US News Brief <span class="cloud-badge">&#9729; CLOUD</span></h1>
  <div class="date">{date_str}</div>
  <div class="meta-bar">
    <span>&#128214; ~30 min read</span><span>&#128272; TOEIC 600+</span><span>&#128451; Politics &middot; Economy &middot; Research</span>
  </div>
</header>
<div class="tips">
  <span class="icon">&#128161;</span>
  <div>
    <strong>Click any word</strong> to look up its Chinese meaning.
    <strong>Click &ldquo;Chinese Translation&rdquo;</strong> under each article for full Chinese text.
    Use the <strong>left sidebar</strong> to browse past editions.
  </div>
</div>
<div class="section">{sections}</div>
<footer>
  <p>&#128218; Auto-generated via NewsAPI + DeepSeek | Cloud pipeline &mdash; GitHub Actions | {date_str}</p>
</footer>
</div>

<div id="word-popover">
  <button class="pop-close" id="pop-close-btn">&times;</button>
  <div class="pop-word" id="pop-word"></div><div class="pop-pos" id="pop-pos"></div><div class="pop-meaning" id="pop-meaning"></div>
</div>
<div id="sel-toolbar">&#128269; Look up "<span id="sel-word"></span>"</div>
<div id="history-panel">
  <div id="history-list"><div id="history-clear">Clear all</div></div>
  <button id="history-toggle" title="Lookup History">&#128214;<span id="history-badge">0</span></button>
</div>

<script>
var DICT = {{{dict_entries}
}};
var popover=document.getElementById('word-popover'),popWord=document.getElementById('pop-word'),popPos=document.getElementById('pop-pos'),popMeaning=document.getElementById('pop-meaning'),popClose=document.getElementById('pop-close-btn'),selToolbar=document.getElementById('sel-toolbar'),selWordSpan=document.getElementById('sel-word'),historyList=document.getElementById('history-list'),historyToggle=document.getElementById('history-toggle'),historyBadge=document.getElementById('history-badge'),popoverTimer=null,activeWordEl=null,lookupHistory=[];
try{{var saved=localStorage.getItem('wb_cloud_v2_history');if(saved)lookupHistory=JSON.parse(saved);updateBadge()}}catch(e){{}}
function norm(w){{return w.trim().toLowerCase().replace(/[.,;:!?()"']+$/g,'').replace(/^[.,;:!?()"']+/g,'').replace(/'s$/g,'')}}
function lookup(raw){{var w=norm(raw);if(!w||w.length<2)return null;if(DICT[w])return{{word:raw.trim(),meaning:DICT[w]}};var o=raw.trim().toLowerCase().replace(/[^\\w\\s-]/g,'');if(DICT[o])return{{word:raw.trim(),meaning:DICT[o]}};return null}}
function showPopover(x,y,word,meaning){{if(popoverTimer){{clearTimeout(popoverTimer);popoverTimer=null}}var m=meaning.match(/^(\\w+\\.?)\\s*(.*)/);popWord.textContent=word;popPos.textContent=m?m[1]:'';popMeaning.innerHTML='<span class="zh">'+(m?m[2]:meaning)+'</span>';var pw=popover.offsetWidth,ph=popover.offsetHeight,l=x+12,t=y-ph-8;if(t<10)t=y+20;if(l+pw>window.innerWidth-10)l=x-pw-12;if(l<10)l=10;popover.style.left=l+'px';popover.style.top=t+'px';popover.classList.add('visible')}}
function hidePopover(){{popover.classList.remove('visible')}}
function addHistory(word,meaning){{lookupHistory=lookupHistory.filter(function(h){{return h.word!==word}});lookupHistory.unshift({{word:word,meaning:meaning}});if(lookupHistory.length>30)lookupHistory.pop();try{{localStorage.setItem('wb_cloud_v2_history',JSON.stringify(lookupHistory))}}catch(e){{}};updateBadge();renderHistory()}}
function updateBadge(){{var c=lookupHistory.length;historyBadge.textContent=c;if(c>0)historyBadge.classList.add('show');else historyBadge.classList.remove('show')}}
function renderHistory(){{historyList.innerHTML='';var cd=document.createElement('div');cd.id='history-clear';cd.textContent='Clear all';cd.onclick=function(){{lookupHistory=[];localStorage.removeItem('wb_cloud_v2_history');updateBadge();renderHistory()}};historyList.appendChild(cd);lookupHistory.forEach(function(h){{var d=document.createElement('div');d.className='h-item';var m=h.meaning.match(/^(\\w+\\.?)\\s*(.*)/);d.innerHTML='<span class="h-word">'+h.word+'</span><span class="h-zh">'+(m?m[2]:h.meaning)+'</span>';historyList.appendChild(d)}})}}
document.querySelectorAll('.news-card p:not(.wim):not(.source), .news-card h3').forEach(function(el){{el.addEventListener('click',function(e){{var sel=window.getSelection();if(sel.toString().trim())return;var range=document.caretRangeFromPoint(e.clientX,e.clientY);if(!range)return;var tn=range.startContainer;if(tn.nodeType!==Node.TEXT_NODE)return;var text=tn.textContent,off=range.startOffset,start=off;while(start>0&&/[\\w'-]/.test(text[start-1]))start--;var end=off;while(end<text.length&&/[\\w'-]/.test(text[end]))end++;var w=text.slice(start,end).trim();if(!w||w.length<2)return;var r=lookup(w);if(!r){{hidePopover();return}}if(activeWordEl)activeWordEl.classList.remove('active');var span=document.createElement('span');span.className='lookup-word active';span.textContent=w;var before=text.slice(0,start),after=text.slice(end);var p=tn.parentNode;p.insertBefore(document.createTextNode(before),tn);p.insertBefore(span,tn);p.insertBefore(document.createTextNode(after),tn);p.removeChild(tn);activeWordEl=span;setTimeout(function(){{if(activeWordEl===span)span.classList.remove('active')}},3000);var rect=span.getBoundingClientRect();showPopover(rect.left+rect.width/2,rect.top,r.word,r.meaning);addHistory(r.word,r.meaning)}})}});
document.addEventListener('mouseup',function(e){{setTimeout(function(){{var sel=window.getSelection();var text=sel.toString().trim();if(!text||text.length<2||text.length>40){{selToolbar.classList.remove('visible');return}}var n=sel.anchorNode,ok=false;while(n){{if(n.classList&&n.classList.contains('news-card')){{ok=true;break}}n=n.parentNode}}if(!ok){{selToolbar.classList.remove('visible');return}}var rect=sel.getRangeAt(0).getBoundingClientRect();selWordSpan.textContent=text;var l=rect.left+rect.width/2-selToolbar.offsetWidth/2,t=rect.bottom+8;if(l<10)l=10;if(l+selToolbar.offsetWidth>window.innerWidth-10)l=window.innerWidth-selToolbar.offsetWidth-10;selToolbar.style.left=l+'px';selToolbar.style.top=t+'px';selToolbar.classList.add('visible');selToolbar._st=text}},10)}});
selToolbar.addEventListener('click',function(){{var text=selToolbar._st;if(!text)return;var pk=text.toLowerCase(),r=null;if(DICT[pk])r={{word:text,meaning:DICT[pk]}};else{{var ws=text.split(/\\s+/);for(var i=0;i<ws.length;i++){{r=lookup(ws[i]);if(r)break}}}}if(r){{var rect=selToolbar.getBoundingClientRect();showPopover(rect.left+rect.width/2,rect.top,r.word,r.meaning);addHistory(r.word,r.meaning)}}selToolbar.classList.remove('visible');window.getSelection().removeAllRanges()}});
document.addEventListener('click',function(e){{if(e.target.closest('.news-card'))return;if(popover.contains(e.target))return;if(e.target===selToolbar||selToolbar.contains(e.target))return;if(e.target.closest('#history-panel')||e.target.closest('#sidebar'))return;popoverTimer=setTimeout(hidePopover,200)}});
popover.addEventListener('mouseenter',function(){{if(popoverTimer){{clearTimeout(popoverTimer);popoverTimer=null}}}});
popover.addEventListener('mouseleave',function(){{popoverTimer=setTimeout(hidePopover,300)}});
popClose.addEventListener('click',function(e){{e.stopPropagation();hidePopover()}});
window.addEventListener('scroll',function(){{hidePopover();selToolbar.classList.remove('visible')}},{{passive:true}});
historyToggle.addEventListener('click',function(e){{e.stopPropagation();renderHistory();historyList.classList.toggle('open')}});
document.addEventListener('click',function(e){{if(!historyList.contains(e.target)&&e.target!==historyToggle&&!historyToggle.contains(e.target))historyList.classList.remove('open')}});
document.addEventListener('keydown',function(e){{if(e.key==='Escape'){{hidePopover();selToolbar.classList.remove('visible');historyList.classList.remove('open')}}}});
// Sidebar archive loader
(function(){{
  var sd=document.getElementById('sidebar-dates');
  fetch('archive/index.json',{{cache:'no-cache'}}).then(function(r){{return r.json()}}).then(function(dates){{
    if(!dates.length){{sd.innerHTML='<p>No archived editions yet.</p>';return}}
    var today='{TODAY}';sd.innerHTML='';
    dates.forEach(function(d){{var a=document.createElement('a');a.className='date-link';a.href='archive/'+d+'.html';a.textContent=d;if(d===today)a.classList.add('active');sd.appendChild(a)}})
  }}).catch(function(){{sd.innerHTML='<p>Archive index loading...</p>'}})
}})();
updateBadge();
</script>
</body>
</html>""".replace("{TODAY}", TODAY)


# ── Archive ─────────────────────────────────────────────
def update_archive(html: str, today: str = TODAY):
    import os as _os
    _os.makedirs(ARCHIVE_DIR, exist_ok=True)
    with open(f"{ARCHIVE_DIR}/{today}.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Archived: {ARCHIVE_DIR}/{today}.html")

    idx_path = f"{ARCHIVE_DIR}/index.json"
    dates = []
    if _os.path.exists(idx_path):
        with open(idx_path) as f:
            dates = json.load(f)
    if today not in dates:
        dates.append(today)
        dates.sort()
    with open(idx_path, "w") as f:
        json.dump(dates, f)
    print(f"  Archive index updated: {len(dates)} entries")


# ── Main ────────────────────────────────────────────────
def main():
    print(f"=== Cloud Pipeline v2 | {datetime.now(BEIJING_TZ).isoformat()} ===")
    all_arts = {}
    for label, cat in CATEGORIES.items():
        print(f"\n[FETCH] {label}")
        raw = fetch_newsapi(cat)
        print(f"  {len(raw)} raw")
        f = filter_articles(raw)
        print(f"  {len(f)} filtered")
        for a in f[:3]: print(f"    [{a['_q']['score']}/8] {a.get('title','?')[:70]}")
        all_arts[label] = f

    summary = "\n\n".join(build_summary(l, all_arts[l]) for l in ["POLITICS","ECONOMY","RESEARCH"])

    print(f"\n[GENERATE] DeepSeek API...")
    data = generate_content(summary)
    print(f"  Articles: {sum(len(data.get(k,[])) for k in ['politics','economy','research'])}")
    print(f"  Dict: {len(data.get('dictionary',{}))} entries")

    print(f"\n[BUILD] HTML...")
    html = build_html(data)
    with open("index.html","w",encoding="utf-8") as f: f.write(html)
    print(f"  index.html ({len(html)/1024:.1f} KB)")

    print(f"\n[ARCHIVE]")
    update_archive(html)
    print(f"\n=== Done ===")

if __name__ == "__main__":
    main()
