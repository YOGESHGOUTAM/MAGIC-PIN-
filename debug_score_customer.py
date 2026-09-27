import sys
import judge_simulator

sys.stdout.reconfigure(encoding='utf-8')

llm = judge_simulator.create_provider()
loader = judge_simulator.DatasetLoader(judge_simulator.DATASET_DIR)
loader.load()
scorer = judge_simulator.LLMScorer(llm, loader)

cat = loader.categories['dentists']
m = loader.merchants['m_001_drmeera_dentist_delhi']
trg = loader.triggers['trg_003_recall_due_priya']
cust = loader.customers['c_001_priya_for_m001']

action = {
    'body': "Hi Priya, Dr. Meera's Dental Clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm for Dental Cleaning @ ₹299. Reply 1 for Wed, 2 for Thu, or tell us a time that works.",
    'cta': 'open_ended',
    'send_as': 'merchant_on_behalf'
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

Customer: {cust.get('identity', {})}

=== BOT'S MESSAGE ===
Body: "{action.get('body', '')}"
CTA: {action.get('cta', 'none')}
Send As: {action.get('send_as', 'vera')}

Score each dimension 0-10 with clear reasoning. Be STRICT."""

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
