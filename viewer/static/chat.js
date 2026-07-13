"use strict";
const $ = s => document.querySelector(s);
let MESSAGES = [];

const SUGGEST = [
  "What can this lab actually simulate?",
  "What does the thermostat coupling constant tau-t do?",
  "Simulate a box of water at 310 K and tell me if the density is right",
  "Run lysozyme in 0.15 M salt for 50 ps and tell me if it stays folded",
];

function esc(s){ return (s||"").replace(/[<>&]/g, c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c])); }
function md(s){
  return esc(s)
    .replace(/```([\s\S]*?)```/g, (m,c)=>`<pre>${c.trim()}</pre>`)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\n/g, "<br>");
}

function addMsg(role, html, tools){
  const d = document.createElement("div");
  d.className = "msg " + (role==="user"?"user":"bot");
  d.innerHTML = `<div class="bubble">${html}</div>` +
    (tools && tools.length ? `<details class="tools"><summary>${tools.length} tool call${tools.length>1?"s":""} — what it actually did</summary>` +
      tools.map(t=>`<div class="tool"><div class="tname">${t.tool}</div>
        <pre class="tin">${esc(JSON.stringify(t.input).slice(0,400))}</pre>
        <pre class="tout">${esc(JSON.stringify(t.output).slice(0,700))}</pre></div>`).join("") + `</details>` : "");
  $("#thread").appendChild(d);
  d.scrollIntoView({behavior:"smooth", block:"end"});
  return d;
}

async function send(text){
  if(!text.trim()) return;
  addMsg("user", md(text));
  MESSAGES.push({role:"user", content:text});
  $("#input").value = "";
  const thinking = addMsg("bot", `<span class="thinking">thinking + calling tools…</span>`);
  $("#send").disabled = true;
  try{
    const r = await fetch("/api/chat", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({messages: MESSAGES})}).then(x=>x.json());
    thinking.remove();
    if(r.error === "no_api_key"){ showNoKey(r.reply); return; }
    addMsg("bot", md(r.reply||"(no reply)"), r.tool_calls);
    if(r.messages) MESSAGES = r.messages;
  }catch(e){
    thinking.remove();
    addMsg("bot", `<span style="color:var(--err)">Request failed: ${esc(e.message)}</span>`);
  }finally{ $("#send").disabled = false; }
}

function showNoKey(msg){
  const el = $("#nokey");
  el.classList.remove("hidden");
  el.innerHTML = `<b>The chatbot needs an API key to think.</b>
    <p>On the lab machine (<code>spark-8d6e</code>) run:</p>
    <pre>export ANTHROPIC_API_KEY=sk-ant-...
# then restart:  python viewer/app.py 5057</pre>
    <p class="muted">Everything else — Plan Builder, Evals, the viewer, the scheduler — works without it.</p>`;
}

$("#chat-form").onsubmit = e => { e.preventDefault(); send($("#input").value); };
$("#input").addEventListener("keydown", e => {
  if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); send($("#input").value); }
});
$("#suggest").innerHTML = SUGGEST.map(s=>`<button class="pchip" data-s="${esc(s)}">${esc(s)}</button>`).join("");
document.querySelectorAll("#suggest .pchip").forEach(b=>b.onclick=()=>send(b.dataset.s));

fetch("/api/chat/status").then(r=>r.json()).then(d=>{
  $("#chat-sub").textContent = d.ready
    ? `driving the real tools · ${d.model}`
    : "needs ANTHROPIC_API_KEY — see below";
  if(!d.ready) showNoKey();
});
