# BUILD_GUIDE.md — NykaaAssist Kaise Banaye (Step-by-Step, Hinglish)

> **Ye file kya hain:** Ye tera "samjho aur banao" guide hai — concepts simple bhasha mein,
> aur har task ka roadmap. Ye copy-paste karne wali code file **nahi** hai — brief mein
> clearly mana hain AI code-gen tools use karne se, toh actual `.py` files tujhe khud
> likhni hain. Ye guide sirf tera dimaag clear karega ki **kya banana hain, kis order
> mein, aur kaise sochna hain.**
>
> Tip: Har task ke baad khud se bolna — "maine ye kyu likha?" — agar answer nahi aata,
> matlab abhi samjha nahi, wapas is file ko padh.

---

## 0. Sabse Pehle — 10 Min Mein Saare Concepts Samajh Lo

Isko skip mat karna. Agar ye 8 cheezein samajh gaya, poora project easy lagega.

| Concept | Simple Matlab |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Pehle knowledge base mein se relevant text dhundo ("retrieval"), phir usi text ke base pe answer banao. Bina retrieval ke answer = guessing = mana hain. |
| **Embedding** | Text ko numbers ki list (vector) mein convert karna, taaki computer "similarity" (matlab kitna similar hain 2 sentences) calculate kar sake. Jaise "return policy" aur "refund kab milega" — dono ke embeddings close honge, kyuki topic similar hain. |
| **Chunking** | Bade document ko chhote pieces mein todna, taaki embedding aur search accurate ho. Do tarike: (1) fixed-size — har chunk N words ka, thoda overlap ke saath. (2) sentence-based — har chunk = ek ya do complete sentences. |
| **Vector Database (ChromaDB)** | Ek jagah jaha saare embeddings store hote hain, aur jab query aati hain, wo query ke embedding ke sabse "close" chunks nikal ke deta hain (top-k retrieval). |
| **Cosine Similarity** | Do embeddings kitna "similar direction" mein hain, uska score — 1 ke close matlab bahut similar, 0 ke close matlab unrelated. Isi score se decide karte hain "I don't know" bolna hain ya nahi. |
| **LangGraph** | Ek flowchart engine — tu nodes (steps) banata hain aur edges (connections) define karta hain ki ek step ke baad kaunsa step chalega. Conditional edge = "agar X toh yaha jao, agar Y toh wahan jao" — jaise humare case mein "policy question → RAG" vs "order question → tool". |
| **MCP (Model Context Protocol)** | Ek standard tareeka tools ko expose karne ka, taaki koi bhi alag client (alag process/program) usi tool ko call kar sake bina tumhare agent code ke andar ghuse. Socho jaise ek REST API, but AI tools ke liye standardized. |
| **Checkpointing** | Agent ka progress beech mein bhi save hota rehta hain (SQLite file mein). Agar process crash ho jaye ya beech mein ruk jaye, wapas wahi se continue kar sakte ho jahan chhoda tha — poora se dobara nahi karna padta. |

Agar ye table samajh aa gaya — tu ready hain.

---

## 1. Setup (Day 0, isse pehle kuch mat karo)

1. Python 3.11 install karo, virtual environment banao:
   ```
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```
2. Ek `requirements.txt` banao (list `TRD.md` §10 mein hain) aur install karo.
3. `MOCK_LLM=1` jaisa ek `.env` file banao (git-ignore karna, commit mat karna).
4. Empty GitHub repo banao, `README.md`/`PRD.md`/etc. is folder mein already hain — inko
   repo ke root mein daal do.

**Test karo:** `python -c "import langgraph, chromadb, sentence_transformers, fastapi"`
— agar error nahi aaya, setup sahi hain.

---

## 2. Task-by-Task Roadmap

Har task ke liye 4 cheezein: **Kya banana hain → Kaise sochna hain → Khud test kaise
karoge → Common galti jo log karte hain.**

### Task 1 — Dataset Generator (`dataset.py`)

- **Kya banana:** `random.seed(<tera number>)` fix karo, phir loop chala ke 40+ order
  records banao — har record mein `record_id`, `category`, `status`, `order_value_inr`,
  `days_since_created`, `delayed_shipment`.
- **Kaise sochna:** `category` aur `status` ke liye Python ka `random.choices()` use karo
  weights ke saath — weights tu khud decide kar (e.g. Apparel zyada common ho sakta hain,
  Electronics kam). `delayed_shipment` bhi weighted random bool ho — pehle 20% weight try
  karo, phir actual % nikal ke dekho.
- **Test:** Script chalane ke baad print karo — category counts, status counts, aur
  `delayed_shipment` ka %. Agar % 10-30 ke bahar hain, **seed ya weight badlo aur phir se
  chalao** — kabhi bhi manually kisi record ko edit mat karna, brief mein clearly mana
  hain.
- **Common galti:** Log log seed change karne ke bajaye ek-do record hand-edit kar dete
  hain — grader ye pakad leta hain kyunki phir dataset reproducible nahi rehta.

### Task 2 — Knowledge Base (`knowledge_base/*.md`)

- **Kya banana:** 12 topics diye hain brief mein (return window, COD refund, delivery SLA,
  reverse pickup, warranty, cancellation, loyalty points, payment retry, size exchange,
  damaged item, international shipping, escalation matrix) — har ek pe apne words mein
  2-5 sentences.
- **Kaise sochna:** Socho tu khud Nykaa ka policy team ho — realistic likho. Copy mat
  karna kahi se, apne words mein original likhna hain (brief explicitly kehta hain).
- **Test:** Har document padh ke khud check karo — kya isme genuinely wo topic cover ho
  raha hain jo required hain?
- **Common galti:** Bahut short (1 line) likh dena — minimum 2 sentences chahiye har doc
  mein.

### Task 3 — Dual Chunking + Embedding + Indexing

- **Kya banana:** Do function — ek fixed-size+overlap chunker, ek sentence-based chunker.
  Dono se saare 12 documents ko chunk karo, phir `sentence-transformers` se embed karo,
  phir do alag ChromaDB collections mein daalo.
- **Kaise sochna:** Fixed-size chunker mein tu decide karta hain chunk size (e.g. 50
  words) aur overlap (e.g. 10 words) — overlap isliye taaki sentence beech mein na kate.
  Sentence-based mein tu Python ki `nltk` ya simple `.split('. ')` type logic se sentences
  todta hain, phir 1-2 sentences ka ek chunk banata hain.
- **Test:** Dono collections mein chunk count print karo, aur ek sample query chala ke
  dekho kya sensible results aa rahe hain.
- **Common galti:** Dono strategies ko ek hi collection mein daal dena — brief mein clearly
  do **separate** collections chahiye.

### Task 4 — Grounded Generation + Threshold

- **Kya banana:** Ek function jo query leke top-k chunks retrieve kare, aur sirf unhi
  chunks ke text se answer banaye.
- **Kaise sochna (threshold calibration — important):** Pehle 3 in-scope queries chalao
  (jaise "return window for footwear?"), unka top-1 similarity note karo. Phir 2
  out-of-scope queries chalao (jaise "what's the weather today?"), unka bhi similarity
  note karo. Do clusters dikhenge — ek high (in-scope), ek low (out-of-scope). Threshold
  in dono ke **beech mein** rakho. Kabhi bhi 0.5/0.6/0.7 seedha mat use karna bina test
  kiye — brief explicitly disallow karta hain.
- **Test:** 5 in-scope + 1 out-of-scope query se demo chalao — out-of-scope wala fallback
  trigger karna chahiye.
- **Common galti:** Threshold ko tutorial se copy karna bina apne data pe test kiye.

### Task 5 — Evaluate Both Chunking Strategies

- **Kya banana:** Same 5 queries jo Task 4 mein use ki, un par Precision@3 aur Recall@3
  nikalo — dono collections ke liye alag-alag.
- **Kaise sochna:** Precision@3 = (retrieved top-3 mein se kitne relevant hain) / 3.
  Recall@3 = (retrieved top-3 mein se kitne relevant hain) / (total relevant docs jo exist
  karte hain us query ke liye). Chunk-level results ko **parent document** pe map karo
  aur dedupe karo pehle scoring karne se pehle.
- **Test:** Per-query numbers print karo dono collections ke liye, phir 2-3 lines mein
  decide karo kaunsi strategy better hain — apne numbers ka reference dete hue.
- **Common galti:** Sirf ek collection ka result dikhana, ya numbers cite kiye bina "mujhe
  ye better lagta hain" bol dena.

### Task 6 — `check_order_status` + Escalation Score

- **Kya banana:** Function jo `record_id` leke dataset se match kare, aur ek
  `escalation_score` calculate kare (0 se 1 ke beech).
- **Kaise sochna:** Formula design karo jo `delayed_shipment` (0 ya 1) aur normalized
  `days_since_created` (recency) dono ko combine kare — jaise
  `score = 0.6*delayed + 0.4*(days/max_days)`. Weights tu justify karega apne dataset ke
  hisaab se. Threshold decide karne ke liye dataset ke `days_since_created` ka distribution
  dekho — e.g. 80th percentile pe threshold rakho, aur bata **kyu** (percentile calculate
  karke dikhao).
- **Test:** 2-3 alag record_id try karo, dekho score sensible hain (jyada delayed +
  purana order = high score).
- **Common galti:** Sirf `if delayed: return 1 else: return 0` likh dena — brief kehta
  hain "not a bare boolean OR", genuine weighted formula chahiye.

### Task 7 — LangGraph Agent Graph

- **Kya banana:** Kam se kam 4 nodes ka graph, aur ek conditional edge jo query dekh ke
  decide kare RAG tool use karna hain ya order tool.
- **Kaise sochna:** Nodes: `input_guardrails → route → {rag_node | order_node} →
  output_groundedness_check`. Routing logic simple rakh sakte ho — agar query mein
  order-id pattern (jaise "NYK-" se start) ya "order status" jaisa phrase hain toh order
  tool, warna RAG.
- **Test:** Do alag queries chalao — ek policy wali, ek order wali — dono alag route pe
  jaate hue dikhao (print karke ya log karke).
- **Common galti:** Conditional edge fake honi — matlab hamesha ek hi path chalna, dusra
  kabhi trigger na hona. Dono paths ko genuinely test karo.

### Task 8 — Persisted Memory

- **Kya banana:** Conversation history ko JSON file mein save karo, `thread_id` ke basis
  pe.
- **Kaise sochna:** Har turn ke baad, us `thread_id` ki history list mein naya
  message+response append karo aur file mein wapas likho. Naye turn pe pehle load karo.
- **Test:** Same `thread_id` se 2 messages bhejo, dikhao doosre message mein pehle wale
  ka context yaad hain. Phir ek naya `thread_id` use karo, dikhao wahan history empty
  hain.
- **Common galti:** Memory ko sirf in-memory Python variable mein rakhna (process restart
  hote hi gayab) — file/DB mein persist hona chahiye.

### Task 9 — Structured Output Schema

- **Kya banana:** JSON Schema define karo (`TRD.md` §3.3 mein example hain), aur har
  response ko usse validate karo code mein (`jsonschema` library use kar sakte ho).
- **Test:** Ek intentionally galat-shaped response banao, dikhao validation fail hota
  hain — phir sahi response se dikhao pass hota hain.

### Task 10 — Guardrails

- **Kya banana:** (a) PII masking — regex se phone number aur card last-4 dhundo aur mask
  karo. (b) Prompt-injection detection — kuch fixed phrases match karo ("ignore previous
  instructions" type). (c) Output groundedness check — agar generated answer retrieved
  context se support nahi ho raha, fallback do.
- **Test:** Teeno guardrails ko ek-ek deliberate test case se trigger karke dikhao —
  jaise fake phone number bhejo, dekho mask hua; injection phrase bhejo, dekho reject
  hua; out-of-context answer force karo, dekho fallback aaya.

### Task 11 — FastAPI Deployment

- **Kya banana:** `POST /ask` aur `POST /add-document` — Pydantic models ke saath
  request/response.
- **Test:** `uvicorn` chalao, `curl` ya Postman se dono endpoints test karo.

### Task 12 — Structured Logging

- **Kya banana:** Har request ko ek JSON-line likhna kisi `.jsonl` file mein — trace_id,
  timestamp, duration, masked query.
- **Kaise sochna:** Jo masking function Task 10 mein banaya, wahi function yaha bhi call
  karo logging se pehle — do alag masking logic mat banana, warna PII kahi se leak ho
  sakta hain.
- **Test:** Ek request bhejo jisme phone number ho, log file khol ke dekho ki wo masked
  hain, raw nahi.

### Task 13 — RAG-Triad Evaluation (15 Queries)

- **Kya banana:** 15 test queries ka set (har KB topic se kam se kam 1, plus 2
  out-of-scope), aur teen scores har query ke liye: context relevance, groundedness,
  answer relevance.
- **Kaise sochna (MOCK_LLM mein):** Chunk aur answer ke beech overlap/similarity se score
  nikalo — rule-based function likho, "black box" nahi honi chahiye, taaki explain kar
  sako scoring kaise hui.
- **Test:** Saare 15 queries ka result table print karo, plus teeno scores ka average.

### Task 14 — MCP Server + Client

- **Kya banana:** `fastmcp` (naam dhyan se — `fastmcp`, `fast-mcp` nahi) se
  `check_order_status` ko ek MCP tool bana ke local server chalao. Ek **alag** Python file
  mein client banao jo us server se connect karke tool call kare.
- **Test:** Client se 2 alag record_id ke liye call karo, standardized MCP response print
  karo dono ke liye.

### Task 15 — SQLite Checkpointing

- **Kya banana:** `langgraph-checkpoint-sqlite` install karo, graph ko SQLite
  checkpointer ke saath configure karo, `thread_id` ke basis pe.
- **Test:** Ek run chalao jo 2 nodes tak jaye, phir process ko manually rok do (before
  remaining nodes). Phir same `thread_id` se resume karo — print karke dikhao ki pehle 2
  nodes **loaded** hue (re-execute nahi hue), aur baaki nodes ab execute hue.

### Task 16 — Timeouts + Retries

- **Kya banana:** (a) Ek node pe per-node timeout. (b) Poore graph pe ek global timeout.
  (c) Ek simulated flaky node jo pehle 2 baar fail ho (counter use karke), phir succeed —
  exponential backoff retry ke saath (max attempts, initial interval, max interval,
  jitter state karo).
- **Test:** Teeno scenario alag-alag demonstrate karo — retry recover hote hue, per-node
  timeout clean error dete hue (hang nahi), global timeout poore run ko cancel karte hue.

---

## 3. Suggested Daily Flow (`PHASES.md` follow karo)

Roz raat ko ye 3 sawaal khud se poocho:
1. Aaj jo bana, wo **actually chal raha hain** ya sirf likha hain?
2. Terminal ka output/transcript **save kiya** `transcripts/` folder mein?
3. Koi number (threshold, percentile, score) nikla toh `README.md` §6 mein update kiya?

Agar teeno "haan" hain, agle din aage badho. Agar nahi, wahi din thoda extra do — aage ka
kaam pichhle pe depend karta hain (Part 2 depends Part 1's recommendation pe, etc.)

---

## 4. Jahan Atak Jao Wahan Kya Karo

- Concept samajh nahi aa raha → is guide ke §0 wapas padho, phir official docs check karo
  (`python.langchain.com`, `fastmcp` docs, `chromadb` docs — brief allow karta hain ye
  refer karna).
- Code likhte waqt confuse ho → chhote function se start karo, print statements daal ke
  step-by-step check karo kya ho raha hain — pura function ek saath mat likhna.
- Number sensible nahi lag raha (jaise % 10-30 ke bahar) → seed/weight badlo, dobara
  generate karo, hand-edit kabhi nahi.

## 5. Final Submission Checklist

- [ ] Saare 16 tasks ke transcripts `transcripts/` mein saved
- [ ] `README.md` §6 ke saare `TBD` values actual numbers se replace
- [ ] `requirements.txt` complete aur versions pinned
- [ ] `.env` git-ignored, koi secret commit nahi hua
- [ ] Poora project `MOCK_LLM=1` pe bina kisi API key ke chalta hain — end-to-end khud
      test karo ek baar fresh clone karke
- [ ] Repo public hain aur ek hi link submit karna hain
