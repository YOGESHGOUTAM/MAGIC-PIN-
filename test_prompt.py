import json
import urllib.request
from pathlib import Path

key = open("api_keys").read().splitlines()[1].strip()

# Load dentists + Dr. Meera + research digest trigger
cat = json.load(open("dataset/categories/dentists.json"))
merchants = json.load(open("dataset/merchants_seed.json"))["merchants"]
m = [x for x in merchants if x["merchant_id"] == "m_001_drmeera_dentist_delhi"][0]
triggers = json.load(open("dataset/triggers_seed.json"))["triggers"]
trg = [x for x in triggers if x["id"] == "trg_001_research_digest_dentists"][0]

prompt = f"""You are Vera, magicpin's merchant AI assistant in India.
Compose a high-converting WhatsApp message based strictly on the 4 contexts below.

=== CATEGORY CONTEXT ===
{json.dumps(cat, indent=2)}

=== MERCHANT CONTEXT ===
{json.dumps(m, indent=2)}

=== TRIGGER CONTEXT ===
{json.dumps(trg, indent=2)}

=== CUSTOMER CONTEXT ===
None (Merchant-facing)

CRITICAL RULES:
1. Specificity: Anchor on concrete facts (numbers, dates, source citations like JIDA Oct 2026 p.14, prices).
2. Category Voice: Respect category tone, terminology, and strictly AVOID all taboo words (no guaranteed, no cure, no 100% safe).
3. Merchant Fit: Use correct owner/merchant name, location, and real data.
4. Trigger Relevance: Immediately communicate WHY NOW based on the trigger.
5. Engagement Compulsion: End with a single, clear, low-friction binary call-to-action (e.g., Reply YES).
6. No fabrication: Never invent data or citations not present in the contexts.
7. Concise: 35-65 words, natural WhatsApp format.

Respond ONLY with valid JSON:
{{
  "body": "<the WhatsApp message text>",
  "cta": "binary",
  "send_as": "vera",
  "suppression_key": "{trg.get('suppression_key', '')}",
  "rationale": "<1-2 sentence strategy explanation>"
}}"""

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={key}"
body = json.dumps({
    "contents": [{"parts": [{"text": prompt}]}],
    "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}
}).encode("utf-8")

req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=15)
data = json.loads(resp.read().decode("utf-8"))
text = data["candidates"][0]["content"]["parts"][0]["text"]
print("Gemini response:")
print(text)
