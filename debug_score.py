import judge_simulator

llm = judge_simulator.create_provider()
loader = judge_simulator.DatasetLoader(judge_simulator.DATASET_DIR)
loader.load()
scorer = judge_simulator.LLMScorer(llm, loader)

cat = loader.categories['dentists']
m = loader.merchants['m_001_drmeera_dentist_delhi']
trg = loader.triggers['trg_001_research_digest_dentists']

action = {
    'body': "Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult cohort in Lajpat Nagar — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Your Dental Cleaning @ ₹299 offer pairs well with this recall protocol. Want me to pull the 2-min abstract and draft a patient WhatsApp? — JIDA Oct 2026 p.14",
    'cta': 'binary',
    'send_as': 'vera'
}

prompt = f"""SCORE THIS MESSAGE:

=== CONTEXT PROVIDED TO BOT ===
Category: {cat.get('slug', 'unknown')}
Voice: {cat.get('voice', {}).get('tone', 'unknown')}
Taboos: {cat.get('voice', {}).get('vocab_taboo', [])[:5]}

Merchant: {m.get('identity', {}).get('name', 'unknown')}
Owner: {m.get('identity', {}).get('owner_first_name', 'unknown')}
Locality: {m.get('identity', {}).get('locality', 'unknown')}
Languages: {m.get('identity', {}).get('languages', [])}
Performance: views={m.get('performance', {}).get('views', '?')}, calls={m.get('performance', {}).get('calls', '?')}, ctr={m.get('performance', {}).get('ctr', '?')}
Signals: {m.get('signals', [])}
Active Offers: {[o.get('title') for o in m.get('offers', []) if o.get('status') == 'active']}

Trigger Kind: {trg.get('kind', 'unknown')}
Trigger Payload: {trg.get('payload', {})}
Trigger Urgency: {trg.get('urgency', '?')}

Customer: None (merchant-facing)

=== BOT'S MESSAGE ===
Body: "{action.get('body', '')}"
CTA: {action.get('cta', 'none')}
Send As: {action.get('send_as', 'vera')}

Score each dimension 0-10 with clear reasoning. Be STRICT."""

try:
    resp = llm.complete(prompt, scorer.SYSTEM)
    print("RAW RESPONSE:")
    print(resp)
    parsed = scorer._parse_response(resp, action)
    print("PARSED TOTAL:", parsed.total, "/ 50")
    print("Specificity:", parsed.specificity, "-", parsed.specificity_reason)
    print("Category Fit:", parsed.category_fit, "-", parsed.category_fit_reason)
    print("Merchant Fit:", parsed.merchant_fit, "-", parsed.merchant_fit_reason)
    print("Decision Quality:", parsed.decision_quality, "-", parsed.decision_quality_reason)
    print("Engagement:", parsed.engagement_compulsion, "-", parsed.engagement_reason)
except Exception as e:
    print("Exception:", e)
