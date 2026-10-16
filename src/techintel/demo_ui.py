"""No-build browser demonstration. All user/model strings use textContent, not innerHTML."""
DEMO_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PatentPulse-Agent demo</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#152a3b;background:#f4f7fa}
main{max-width:980px;margin:34px auto;padding:0 18px}h1{letter-spacing:-.04em}
small{color:#586b78}article{background:#fff;border:1px solid #dce5ec;border-radius:12px;padding:20px;margin:18px 0}
label{display:block;font-weight:600;margin:12px 0 6px}textarea,input,select{box-sizing:border-box;width:100%;padding:10px;border:1px solid #b8cbd7;border-radius:7px;font:inherit}
textarea{min-height:105px}button{background:#153f66;color:white;border:0;border-radius:8px;padding:11px 16px;font:inherit;font-weight:600;margin-top:12px;cursor:pointer}
pre{white-space:pre-wrap;word-break:break-word;background:#eef3f7;border-radius:8px;padding:15px;max-height:400px;overflow-y:auto}
.meta{font-size:13px;color:#52687a} .warn{background:#fff8de;border:1px solid #e4cf78;padding:12px;border-radius:8px}
</style></head><body><main>
<h1>PatentPulse-Agent</h1><p>Evidence-first patent research prototype · <a href="/docs">API documentation</a></p>
<div class="warn">Curated records are human paraphrases of identifiable real patent publications, not full patent texts. Synthetic records are fictional teaching data. This service does not assess novelty, infringement or legal status.</div>
<article><h2>1 · Patent research search</h2>
<label for="source">Source</label><select id="source"><option value="curated">Real-patent source notes (3)</option><option value="local">Synthetic teaching records (4)</option></select>
<label for="query">Technical query</label><input id="query" value="robotic gripper fingertip tactile sensing" maxlength="200"/>
<label for="retrieval">Retriever</label><select id="retrieval"><option value="hybrid">Word+character TF-IDF</option><option value="bm25">BM25</option><option value="word">Word TF-IDF</option></select>
<button id="search">Search with sources</button><pre id="searchresult">Search results will appear here.</pre></article>
<article><h2>2 · Parse patent claims</h2><small>Handles basic English numbered claims and common dependency lists/ranges.</small>
<label for="claims">Claims</label><textarea id="claims">1. A robotic gripper including a tactile sensor.
2. The robotic gripper of claim 1 with a motion controller.
3. The gripper of claims 1 to 2 with a proximity sensor.</textarea>
<button id="parse">Parse dependency graph</button><pre id="parseresult">Dependency graph will appear here.</pre></article>
<article><h2>3 · Compare claim terms</h2><small>Lexical coverage only — not a patent-law conclusion.</small>
<label for="reference">Reference claim</label><textarea id="reference">1. A robotic gripper with a tactile sensor and camera.</textarea>
<button id="compare">Compare against target above</button><pre id="compareresult">Comparison chart will appear here.</pre></article>
<p class="meta">Code and tests: see project README · Model-driven function calling is separately enabled with an API key; this page uses deterministic local APIs.</p>
</main><script>
async function submit(url, payload, outputId){
  const out=document.getElementById(outputId); out.textContent='Running…';
  try{ const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await r.json(); out.textContent=JSON.stringify(data,null,2);
    if(!r.ok) out.textContent='HTTP '+r.status+'\n'+out.textContent;
  }catch(e){out.textContent='Request failed: '+String(e)}
}
document.getElementById('search').onclick=()=>submit('/patents/search',{
query:document.getElementById('query').value,source:document.getElementById('source').value,
retrieval:document.getElementById('retrieval').value,k:3},'searchresult');
document.getElementById('parse').onclick=()=>submit('/patents/claims/parse',{
claims_text:document.getElementById('claims').value},'parseresult');
document.getElementById('compare').onclick=()=>submit('/patents/claims/compare',{
target_claims:document.getElementById('claims').value,reference_claims:document.getElementById('reference').value},'compareresult');
</script></body></html>'''
