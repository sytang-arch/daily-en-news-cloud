#!/usr/bin/env python3
"""
Daily US News Brief — Cloud Pipeline
Fetches news via NewsAPI → generates TOEIC 600+ HTML via DeepSeek → outputs index.html
"""
import os, json, re, sys, hashlib
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen
from urllib.error import URLError

# ── Config ──────────────────────────────────────────────
NEWSAPI_KEY  = os.environ["NEWSAPI_KEY"]
DEEPSEEK_KEY = os.environ["DEEPSEEK_KEY"]
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
OUTPUT_FILE  = "index.html"
BEIJING_TZ   = timezone(timedelta(hours=8))

CATEGORIES = {
    "POLITICS":  "politics",
    "ECONOMY":   "business",
    "RESEARCH":  "science",
}

# ── Known credible domains ──────────────────────────────
CREDIBLE_DOMAINS = {
    "reuters.com", "apnews.com", "nytimes.com", "wsj.com", "washingtonpost.com",
    "bloomberg.com", "cnbc.com", "bbc.com", "bbc.co.uk", "npr.org", "politico.com",
    "theguardian.com", "usatoday.com", "axios.com", "cbsnews.com", "abcnews.go.com",
    "nbcnews.com", "cnn.com", "time.com", "theatlantic.com", "economist.com",
    "ft.com", "science.org", "nature.com", "sciencedaily.com", "newscientist.com",
    "space.com", "phys.org", "sciencenews.org", "livescience.com",
    "nih.gov", "nasa.gov", "energy.gov", "noaa.gov", "nsf.gov",
}
SUSPECT_DOMAINS = {  # known aggregators, blog farms, AI-generated sites
    "medium.com", "blogspot.com", "wordpress.com", "substack.com",
    "yahoo.com", "aol.com", "msn.com",  # aggregators that rehost content
}

def fetch_newsapi(category: str) -> list[dict]:
    """Fetch top headlines from NewsAPI for a given category."""
    url = (
        f"https://newsapi.org/v2/top-headlines"
        f"?country=us&category={category}&pageSize=15&apiKey={NEWSAPI_KEY}"
    )
    req = Request(url, headers={"User-Agent": "DailyENNews/1.0"})
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except URLError as e:
        print(f"  [ERROR] NewsAPI fetch failed for {category}: {e}")
        return []
    if data.get("status") != "ok":
        print(f"  [WARN] NewsAPI status not ok for {category}: {data.get('message','')}")
        return []
    return data.get("articles", [])

# ── Source quality check ────────────────────────────────
def source_quality(article: dict) -> dict:
    """Rate a single article's source quality. Returns {score, flags, domain}."""
    source_name = (article.get("source", {}) or {}).get("name", "Unknown")
    url = article.get("url", "")
    domain = ""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        pass

    flags = []
    score = 5  # start neutral

    # Known credible source
    if any(cred in domain for cred in CREDIBLE_DOMAINS):
        score += 3
        flags.append("trusted-domain")
    # Suspect source
    if any(sus in domain for sus in SUSPECT_DOMAINS):
        score -= 2
        flags.append("aggregator-source")
    # Unknown .com / .org (not clearly credible, not clearly suspect)
    if not flags and domain:
        score -= 1
        flags.append("unknown-domain")
    # Missing source name
    if source_name == "Unknown" or not source_name:
        score -= 2
        flags.append("no-source-name")
    # Content too short (likely just a headline, not a real article)
    content = (article.get("content") or article.get("description") or "")
    if len(content) < 100:
        score -= 2
        flags.append("thin-content")
    # Missing publication date
    if not article.get("publishedAt"):
        score -= 1
        flags.append("no-pubdate")

    return {"score": max(0, score), "flags": flags, "domain": domain or source_name, "source_name": source_name}


def filter_articles(articles: list[dict], min_score: int = 4) -> list[dict]:
    """Filter and rank articles by source quality."""
    rated = []
    for a in articles:
        q = source_quality(a)
        a["_quality"] = q
        rated.append(a)
    rated.sort(key=lambda a: a["_quality"]["score"], reverse=True)
    return [a for a in rated if a["_quality"]["score"] >= min_score][:6]


def build_news_summary(category: str, articles: list[dict]) -> str:
    """Build a compact text summary of articles for the LLM."""
    lines = [f"=== {category} NEWS (top {len(articles)} articles) ==="]
    for i, a in enumerate(articles[:6], 1):
        title = a.get("title", "No title")
        desc  = a.get("description") or a.get("content") or "No description available"
        src   = a.get("source", {}).get("name", "Unknown")
        date  = a.get("publishedAt", "")[:10]
        q     = a.get("_quality", {})
        lines.append(
            f"[{i}] {title}\n"
            f"    Source: {src} | Date: {date} | Quality: {q.get('score','?')}/8\n"
            f"    {desc}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an expert English teacher and news editor. Your task is to create a daily English news reading page for a Chinese student at TOEIC ~600 level (CEFR B1).

## OUTPUT FORMAT
You MUST output ONLY valid JSON with this structure:
{
  "date": "Wednesday, July 29, 2026",
  "politics": [{"title":"...", "paragraphs":["...","...","..."], "source":"..."}, ...],
  "economy": [{"title":"...", "paragraphs":["...","...","..."], "what_it_means":"...", "source":"..."}, ...],
  "research": [{"title":"...", "paragraphs":["...","...","..."], "source":"..."}, ...],
  "dictionary": {"word": "pos. 中文释义", ...}
}

## CONTENT REQUIREMENTS
- 2-3 articles per category.
- English level B1+ to B2. Short paragraphs (2-4 sentences each). Each article must have exactly 3 paragraphs.
- Every technical or domain-specific term (science, law, finance) MUST be explained in plain English within the text using parentheses.
- Economic articles MUST include a "what_it_means" field — a 1-sentence plain-English explanation of why this matters to ordinary people.
- Source line format: "Source: {SourceName} — {Date}"
- The dictionary MUST cover EVERY non-trivial word used in all articles (nouns, verbs, adjectives, adverbs). Format: "word": "pos. 中文释义".

## QUALITY REVIEW (perform internally before output)
1. SOURCE CHECK: If a source is not from a well-known legitimate news outlet, either drop the article or note "via [original source]" in the source line.
2. FRESHNESS: All articles must be within the last 2 days. Drop anything older.
3. BALANCE: No single person or entity should dominate the entire page. Include institutional stories (Congress, courts, agencies) where possible.
4. DIFFICULTY: Scan for C1+ vocabulary (e.g., "ephemeral", "concomitant") and replace with B1-B2 equivalents.
5. TERMINOLOGY: Ensure every field-specific term has an in-text explanation in parentheses.
"""


def call_deepseek(news_text: str) -> dict:
    """Call DeepSeek API to generate TOEIC-level news content."""
    user_msg = (
        f"Today is {datetime.now(BEIJING_TZ).strftime('%A, %B %d, %Y')}.\n\n"
        f"Here are today's US news articles. Generate the TOEIC-level reading page.\n\n"
        f"{news_text}"
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.7,
        "max_tokens": 6000,
        "response_format": {"type": "json_object"},
    }
    req = Request(
        DEEPSEEK_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_KEY}",
        },
    )
    try:
        with urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
    except URLError as e:
        print(f"[ERROR] DeepSeek API call failed: {e}")
        sys.exit(1)

    content = result["choices"][0]["message"]["content"]
    # Parse JSON from response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Sometimes the response has markdown wrapping
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        return json.loads(cleaned)


# ── HTML Generation ──────────────────────────────────────
def build_html(data: dict) -> str:
    """Build the full HTML page from LLM-generated data."""
    date_str = data["date"]
    dict_entries = ",\n".join(
        f'    "{k}":"{v}"' for k, v in data.get("dictionary", {}).items()
    )

    sections_html = ""

    # Politics
    sections_html += '<div class="section-title politics">&#127987; POLITICS</div>\n'
    for i, art in enumerate(data.get("politics", []), 1):
        paras = "".join(f"<p>{p}</p>\n" for p in art["paragraphs"])
        sections_html += f"""<div class="news-card">
    <h3>{i}. {art["title"]}</h3>
    {paras}    <p class="source">{art["source"]}</p>
  </div>
"""

    # Economy
    sections_html += '<div class="section-title economy">&#128176; ECONOMY</div>\n'
    for i, art in enumerate(data.get("economy", []), 1):
        paras = "".join(f"<p>{p}</p>\n" for p in art["paragraphs"])
        wim = art.get("what_it_means", "")
        wim_html = f'<p style="color:#8a6d20;font-size:0.88rem;padding:6px 0 0;">&#128161; <strong>What this means for you:</strong> {wim}</p>\n' if wim else ""
        sections_html += f"""<div class="news-card">
    <h3>{len(data.get("politics",[]))+i}. {art["title"]}</h3>
    {paras}{wim_html}    <p class="source">{art["source"]}</p>
  </div>
"""

    # Research
    offset = len(data.get("politics", [])) + len(data.get("economy", []))
    sections_html += '<div class="section-title research">&#128300; RESEARCH &amp; SCIENCE</div>\n'
    for i, art in enumerate(data.get("research", []), 1):
        paras = "".join(f"<p>{p}</p>\n" for p in art["paragraphs"])
        sections_html += f"""<div class="news-card">
    <h3>{offset+i}. {art["title"]}</h3>
    {paras}    <p class="source">{art["source"]}</p>
  </div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Today's US News — {date_str} | English Reading Practice</title>
<style>
  :root {{
    --bg: #faf9f7; --card-bg: #ffffff; --text: #2d2d2d; --text-light: #5a5a5a;
    --accent-politics: #b22234; --accent-economy: #1a5f8a; --accent-research: #2d7d46;
    --accent-gold: #b7933a; --border: #e8e4df; --tag-bg: #f0ede8;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-lg: 0 12px 40px rgba(0,0,0,0.12); --radius: 10px;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Georgia', 'Times New Roman', serif; background: var(--bg); color: var(--text);
    line-height: 1.8; max-width: 780px; margin: 0 auto; padding: 32px 20px 60px; cursor: default;
  }}
  body.word-lookup-active .news-card p, body.word-lookup-active .news-card h3 {{ cursor: pointer; }}
  header {{ text-align: center; padding: 36px 0 28px; border-bottom: 2px solid var(--border); margin-bottom: 32px; }}
  header h1 {{ font-size: 1.75rem; font-weight: 700; letter-spacing: -0.3px; margin-bottom: 6px; color: #1a1a1a; }}
  header .date {{ font-size: 0.9rem; color: var(--text-light); }}
  .meta-bar {{ display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; margin-top: 14px; font-size: 0.82rem; color: var(--text-light); }}
  .meta-bar span {{ background: var(--tag-bg); padding: 4px 14px; border-radius: 20px; }}
  .section {{ margin-bottom: 40px; }}
  .section-title {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 16px; padding-bottom: 6px; border-bottom: 3px solid; display: flex; align-items: center; gap: 8px; }}
  .section-title.politics {{ color: var(--accent-politics); border-color: var(--accent-politics); }}
  .section-title.economy  {{ color: var(--accent-economy);  border-color: var(--accent-economy); }}
  .section-title.research {{ color: var(--accent-research); border-color: var(--accent-research); }}
  .news-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px 26px; margin-bottom: 16px; box-shadow: var(--shadow); transition: box-shadow 0.3s ease, border-color 0.3s ease; }}
  .news-card:hover {{ box-shadow: 0 3px 12px rgba(0,0,0,0.08); border-color: #d8d2c8; }}
  .news-card h3 {{ font-size: 1.08rem; font-weight: 700; margin-bottom: 10px; line-height: 1.45; }}
  .news-card p {{ font-size: 0.95rem; color: var(--text-light); margin-bottom: 12px; text-align: justify; }}
  .news-card .source {{ font-size: 0.78rem; color: #999; font-style: italic; }}
  .lookup-word {{ cursor: pointer; border-bottom: 1.5px dotted transparent; transition: border-color 0.2s, background 0.2s; border-radius: 2px; padding: 0 1px; }}
  .lookup-word:hover {{ border-bottom-color: var(--accent-gold); background: rgba(183,147,58,0.08); }}
  .lookup-word.active {{ background: rgba(183,147,58,0.18); border-bottom-color: var(--accent-gold); }}
  #word-popover {{
    position: fixed; z-index: 9999; background: #fffef9; border: 1px solid #d8d0b8; border-radius: 12px;
    box-shadow: var(--shadow-lg); padding: 18px 22px; min-width: 200px; max-width: 320px;
    opacity: 0; transform: translateY(8px) scale(0.96);
    transition: opacity 0.2s cubic-bezier(0.16,1,0.3,1), transform 0.25s cubic-bezier(0.16,1,0.3,1);
    pointer-events: none; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; word-break: break-word;
  }}
  #word-popover.visible {{ opacity: 1; transform: translateY(0) scale(1); pointer-events: auto; }}
  #word-popover .pop-word {{ font-size: 1.25rem; font-weight: 700; color: #1a1a1a; margin-bottom: 4px; font-family: Georgia, serif; }}
  #word-popover .pop-pos {{ font-size: 0.75rem; color: #999; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 10px; }}
  #word-popover .pop-meaning {{ font-size: 0.92rem; color: #444; line-height: 1.5; padding: 8px 12px; background: #faf6ec; border-radius: 6px; border-left: 3px solid var(--accent-gold); }}
  #word-popover .pop-meaning .zh {{ color: #6b4c1e; font-weight: 600; }}
  #word-popover .pop-close {{
    position: absolute; top: 8px; right: 10px; width: 24px; height: 24px;
    border: none; background: none; cursor: pointer; font-size: 1rem; color: #bbb; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; transition: color 0.2s, background 0.2s;
  }}
  #word-popover .pop-close:hover {{ color: #666; background: #f0ede8; }}
  #sel-toolbar {{
    position: fixed; z-index: 9998; background: #2d2d2d; color: #fff; border-radius: 8px;
    padding: 6px 14px; font-size: 0.82rem; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    cursor: pointer; opacity: 0; transform: translateY(6px);
    transition: opacity 0.18s ease, transform 0.2s ease; pointer-events: none;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25); white-space: nowrap;
  }}
  #sel-toolbar.visible {{ opacity: 1; transform: translateY(0); pointer-events: auto; }}
  #sel-toolbar:hover {{ background: #3d3d3d; }}
  .tips {{
    background: #f4f9ff; border-radius: var(--radius); padding: 16px 22px; margin-bottom: 32px;
    font-size: 0.88rem; color: #3a5a7c; line-height: 1.7; display: flex; align-items: flex-start; gap: 12px;
  }}
  .tips .icon {{ font-size: 1.4rem; flex-shrink: 0; }}
  .tips strong {{ color: #1a3a5c; }}
  .cloud-badge {{
    display: inline-block; background: #e8f5e9; color: #2d7d46; font-size: 0.7rem;
    padding: 2px 10px; border-radius: 10px; margin-left: 8px; vertical-align: middle;
  }}
  #history-panel {{
    position: fixed; right: 24px; bottom: 80px; z-index: 9990;
    display: flex; flex-direction: column; gap: 6px; align-items: flex-end;
  }}
  #history-toggle {{
    width: 44px; height: 44px; border-radius: 50%; background: #fff; border: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); cursor: pointer; font-size: 1.2rem;
    display: flex; align-items: center; justify-content: center; transition: all 0.2s; position: relative;
  }}
  #history-toggle:hover {{ border-color: var(--accent-gold); box-shadow: 0 4px 16px rgba(0,0,0,0.12); }}
  #history-badge {{
    position: absolute; top: -4px; right: -4px; background: var(--accent-gold); color: #fff;
    font-size: 0.65rem; font-family: -apple-system, sans-serif; min-width: 16px; height: 16px;
    border-radius: 8px; display: flex; align-items: center; justify-content: center; padding: 0 4px;
    opacity: 0; transform: scale(0); transition: all 0.3s cubic-bezier(0.16,1,0.3,1);
  }}
  #history-badge.show {{ opacity: 1; transform: scale(1); }}
  #history-list {{
    background: #fffef9; border: 1px solid #d8d0b8; border-radius: 12px; box-shadow: var(--shadow-lg);
    padding: 14px 16px; max-height: 300px; overflow-y: auto; width: 240px;
    font-family: -apple-system, sans-serif; font-size: 0.82rem; display: none; flex-direction: column; gap: 8px;
  }}
  #history-list.open {{ display: flex; }}
  #history-list .h-item {{
    display: flex; justify-content: space-between; align-items: center; padding: 6px 8px;
    border-radius: 6px; cursor: pointer; transition: background 0.15s; gap: 8px;
  }}
  #history-list .h-item:hover {{ background: #faf6ec; }}
  #history-list .h-word {{ font-weight: 600; color: #1a1a1a; }}
  #history-list .h-zh {{ color: #888; font-size: 0.78rem; }}
  #history-clear {{ text-align: center; font-size: 0.72rem; color: #bbb; cursor: pointer; padding: 4px; border-radius: 4px; transition: color 0.2s; }}
  #history-clear:hover {{ color: #c0392b; }}
  footer {{ text-align: center; padding: 24px 0 0; border-top: 1px solid var(--border); font-size: 0.78rem; color: #aaa; margin-top: 8px; }}
  .news-card p, .news-card h3 {{ -webkit-tap-highlight-color: transparent; }}
  @media (max-width: 600px) {{
    body {{ padding: 20px 14px 40px; }}
    .news-card {{ padding: 16px 18px; }}
    #history-panel {{ right: 12px; bottom: 60px; }}
    #word-popover {{ max-width: 260px; }}
  }}
  @media (pointer: coarse) {{ .news-card p, .news-card h3 {{ cursor: default; }} }}
</style>
</head>
<body>

<header>
  <h1>Today's US News Brief <span class="cloud-badge">&#9729; CLOUD</span></h1>
  <div class="date">{date_str}</div>
  <div class="meta-bar">
    <span>&#128214; ~25 min read</span>
    <span>&#128272; TOEIC 600+ level</span>
    <span>&#128451; Politics &middot; Economy &middot; Research</span>
  </div>
</header>

<div class="tips">
  <span class="icon">&#128161;</span>
  <div>
    <strong>Click any word</strong> to look up its Chinese meaning instantly.
    You can also <strong>select a word or phrase</strong> with your mouse and a lookup button will appear.
    Your looked-up words are saved in the history panel (bottom-right) for later review.
  </div>
</div>

<div class="section">
{sections_html}</div>

<footer>
  <p>&#128218; Auto-generated via NewsAPI + DeepSeek. Content reviewed for source quality and TOEIC level.</p>
  <p>Cloud pipeline — GitHub Actions | {date_str}</p>
</footer>

<div id="word-popover">
  <button class="pop-close" id="pop-close-btn">&times;</button>
  <div class="pop-word" id="pop-word"></div>
  <div class="pop-pos" id="pop-pos"></div>
  <div class="pop-meaning" id="pop-meaning"></div>
</div>
<div id="sel-toolbar">&#128269; Look up "<span id="sel-word"></span>"</div>
<div id="history-panel">
  <div id="history-list"><div id="history-clear">Clear all</div></div>
  <button id="history-toggle" title="Lookup History">&#128214;<span id="history-badge">0</span></button>
</div>

<script>
var DICT = {{
{dict_entries}
}};
var popover=document.getElementById('word-popover'),popWord=document.getElementById('pop-word'),popPos=document.getElementById('pop-pos'),popMeaning=document.getElementById('pop-meaning'),popClose=document.getElementById('pop-close-btn'),selToolbar=document.getElementById('sel-toolbar'),selWordSpan=document.getElementById('sel-word'),historyList=document.getElementById('history-list'),historyToggle=document.getElementById('history-toggle'),historyBadge=document.getElementById('history-badge'),popoverTimer=null,activeWordEl=null,lookupHistory=[];
try{{var saved=localStorage.getItem('wb_cloud_history');if(saved)lookupHistory=JSON.parse(saved);updateBadge()}}catch(e){{}}
function norm(w){{return w.trim().toLowerCase().replace(/[.,;:!?()"']+$/g,'').replace(/^[.,;:!?()"']+/g,'').replace(/'s$/g,'')}}
function lookup(raw){{var w=norm(raw);if(!w||w.length<2)return null;if(DICT[w])return{{word:raw.trim(),meaning:DICT[w]}};var o=raw.trim().toLowerCase().replace(/[^\\w\\s-]/g,'');if(DICT[o])return{{word:raw.trim(),meaning:DICT[o]}};return null}}
function showPopover(x,y,word,meaning){{if(popoverTimer){{clearTimeout(popoverTimer);popoverTimer=null}}var m=meaning.match(/^(\\w+\\.?)\\s*(.*)/);popWord.textContent=word;popPos.textContent=m?m[1]:'';popMeaning.innerHTML='<span class="zh">'+(m?m[2]:meaning)+'</span>';var pw=popover.offsetWidth,ph=popover.offsetHeight,l=x+12,t=y-ph-8;if(t<10)t=y+20;if(l+pw>window.innerWidth-10)l=x-pw-12;if(l<10)l=10;popover.style.left=l+'px';popover.style.top=t+'px';popover.classList.add('visible')}}
function hidePopover(){{popover.classList.remove('visible')}}
function addHistory(word,meaning){{lookupHistory=lookupHistory.filter(function(h){{return h.word!==word}});lookupHistory.unshift({{word:word,meaning:meaning}});if(lookupHistory.length>30)lookupHistory.pop();try{{localStorage.setItem('wb_cloud_history',JSON.stringify(lookupHistory))}}catch(e){{}};updateBadge();renderHistory()}}
function updateBadge(){{var c=lookupHistory.length;historyBadge.textContent=c;if(c>0)historyBadge.classList.add('show');else historyBadge.classList.remove('show')}}
function renderHistory(){{historyList.innerHTML='';var cd=document.createElement('div');cd.id='history-clear';cd.textContent='Clear all';cd.onclick=function(){{lookupHistory=[];localStorage.removeItem('wb_cloud_history');updateBadge();renderHistory()}};historyList.appendChild(cd);lookupHistory.forEach(function(h){{var d=document.createElement('div');d.className='h-item';var m=h.meaning.match(/^(\\w+\\.?)\\s*(.*)/);d.innerHTML='<span class="h-word">'+h.word+'</span><span class="h-zh">'+(m?m[2]:h.meaning)+'</span>';historyList.appendChild(d)}})}}
document.querySelectorAll('.news-card p, .news-card h3').forEach(function(el){{el.addEventListener('click',function(e){{var sel=window.getSelection();if(sel.toString().trim())return;var range=document.caretRangeFromPoint(e.clientX,e.clientY);if(!range)return;var tn=range.startContainer;if(tn.nodeType!==Node.TEXT_NODE)return;var text=tn.textContent,off=range.startOffset,start=off;while(start>0&&/[\\w'-]/.test(text[start-1]))start--;var end=off;while(end<text.length&&/[\\w'-]/.test(text[end]))end++;var w=text.slice(start,end).trim();if(!w||w.length<2)return;var r=lookup(w);if(!r){{hidePopover();return}}if(activeWordEl)activeWordEl.classList.remove('active');var span=document.createElement('span');span.className='lookup-word active';span.textContent=w;var before=text.slice(0,start),after=text.slice(end);var p=tn.parentNode;p.insertBefore(document.createTextNode(before),tn);p.insertBefore(span,tn);p.insertBefore(document.createTextNode(after),tn);p.removeChild(tn);activeWordEl=span;setTimeout(function(){{if(activeWordEl===span)span.classList.remove('active')}},3000);var rect=span.getBoundingClientRect();showPopover(rect.left+rect.width/2,rect.top,r.word,r.meaning);addHistory(r.word,r.meaning)}})}});
document.addEventListener('mouseup',function(e){{setTimeout(function(){{var sel=window.getSelection();var text=sel.toString().trim();if(!text||text.length<2||text.length>40){{selToolbar.classList.remove('visible');return}}var n=sel.anchorNode,ok=false;while(n){{if(n.classList&&n.classList.contains('news-card')){{ok=true;break}}n=n.parentNode}}if(!ok){{selToolbar.classList.remove('visible');return}}var rect=sel.getRangeAt(0).getBoundingClientRect();selWordSpan.textContent=text;var l=rect.left+rect.width/2-selToolbar.offsetWidth/2,t=rect.bottom+8;if(l<10)l=10;if(l+selToolbar.offsetWidth>window.innerWidth-10)l=window.innerWidth-selToolbar.offsetWidth-10;selToolbar.style.left=l+'px';selToolbar.style.top=t+'px';selToolbar.classList.add('visible');selToolbar._st=text}},10)}});
selToolbar.addEventListener('click',function(){{var text=selToolbar._st;if(!text)return;var pk=text.toLowerCase(),r=null;if(DICT[pk])r={{word:text,meaning:DICT[pk]}};else{{var ws=text.split(/\\s+/);for(var i=0;i<ws.length;i++){{r=lookup(ws[i]);if(r)break}}}}if(r){{var rect=selToolbar.getBoundingClientRect();showPopover(rect.left+rect.width/2,rect.top,r.word,r.meaning);addHistory(r.word,r.meaning)}}selToolbar.classList.remove('visible');window.getSelection().removeAllRanges()}});
document.addEventListener('click',function(e){{if(e.target.closest('.news-card'))return;if(popover.contains(e.target))return;if(e.target===selToolbar||selToolbar.contains(e.target))return;if(e.target.closest('#history-panel'))return;popoverTimer=setTimeout(hidePopover,200)}});
popover.addEventListener('mouseenter',function(){{if(popoverTimer){{clearTimeout(popoverTimer);popoverTimer=null}}}});
popover.addEventListener('mouseleave',function(){{popoverTimer=setTimeout(hidePopover,300)}});
popClose.addEventListener('click',function(e){{e.stopPropagation();hidePopover()}});
window.addEventListener('scroll',function(){{hidePopover();selToolbar.classList.remove('visible')}},{{passive:true}});
historyToggle.addEventListener('click',function(e){{e.stopPropagation();renderHistory();historyList.classList.toggle('open')}});
document.addEventListener('click',function(e){{if(!historyList.contains(e.target)&&e.target!==historyToggle&&!historyToggle.contains(e.target))historyList.classList.remove('open')}});
document.addEventListener('keydown',function(e){{if(e.key==='Escape'){{hidePopover();selToolbar.classList.remove('visible');historyList.classList.remove('open')}}}});
updateBadge();
</script>
</body>
</html>"""


def main():
    print(f"=== Daily US News Brief — Cloud Pipeline ===")
    print(f"Run time: {datetime.now(BEIJING_TZ).isoformat()}")

    # Step 1: Fetch news
    all_articles = {}
    for label, cat in CATEGORIES.items():
        print(f"\n[FETCH] {label} ({cat})...")
        raw = fetch_newsapi(cat)
        print(f"  Got {len(raw)} raw articles from NewsAPI")
        filtered = filter_articles(raw, min_score=4)
        print(f"  After quality filter: {len(filtered)} articles")
        for a in filtered[:3]:
            q = a["_quality"]
            print(f"    [{q['score']}/8] {a.get('title','?')[:70]}... | {q['source_name']}")
        all_articles[label] = filtered

    # Step 2: Build summary for LLM
    news_summary = ""
    for label in ["POLITICS", "ECONOMY", "RESEARCH"]:
        news_summary += build_news_summary(label, all_articles[label]) + "\n\n"

    # Step 3: Call DeepSeek
    print(f"\n[GENERATE] Calling DeepSeek API...")
    data = call_deepseek(news_summary)
    print(f"  Got {len(data.get('politics',[]))} politics, {len(data.get('economy',[]))} economy, {len(data.get('research',[]))} research articles")
    print(f"  Dictionary: {len(data.get('dictionary',{}))} entries")

    # Step 4: Build HTML
    print(f"\n[BUILD] Generating HTML...")
    html = build_html(data)

    # Step 5: Write
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = len(html) / 1024
    print(f"  Written {OUTPUT_FILE} ({size_kb:.1f} KB)")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
