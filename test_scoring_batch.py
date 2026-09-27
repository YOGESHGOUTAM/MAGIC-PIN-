import sys
import json
import time
import judge_simulator

sys.stdout.reconfigure(encoding='utf-8')

llm = judge_simulator.create_provider()
loader = judge_simulator.DatasetLoader(judge_simulator.DATASET_DIR)
loader.load()
scorer = judge_simulator.LLMScorer(llm, loader)

# Test 1: Dentist Research Digest
cat1 = loader.categories['dentists']
m1 = loader.merchants['m_001_drmeera_dentist_delhi']
trg1 = loader.triggers['trg_001_research_digest_dentists']
a1 = {
    'body': "Dr. Meera, JIDA's Oct issue landed. One item relevant to your adult patients in Lajpat Nagar — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Your Dental Cleaning @ ₹299 offer pairs well with this protocol. Want me to pull the 2-min abstract and draft a patient WhatsApp message for your clinic? — JIDA Oct 2026 p.14",
    'cta': 'binary',
    'send_as': 'vera'
}
s1 = scorer.score(a1, cat1, m1, trg1, None)
print(f"Test 1 (Dentist Digest) Score: {s1.total}/50")
print(f"  Spec: {s1.specificity}, Cat: {s1.category_fit}, Merch: {s1.merchant_fit}, Dec: {s1.decision_quality}, Eng: {s1.engagement_compulsion}")
if s1.penalties:
    print(f"  Penalties: -{s1.penalties}")

time.sleep(4)

# Test 2: Dentist Customer Recall
cust2 = loader.customers['c_001_priya_for_m001']
trg2 = loader.triggers['trg_003_recall_due_priya']
a2 = {
    'body': "Hi Priya, Dr. Meera's Dental Clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm for Dental Cleaning @ ₹299. Reply 1 for Wed, 2 for Thu, or tell us a time that works.",
    'cta': 'open_ended',
    'send_as': 'merchant_on_behalf'
}
s2 = scorer.score(a2, cat1, m1, trg2, cust2)
print(f"Test 2 (Customer Recall) Score: {s2.total}/50")
print(f"  Spec: {s2.specificity}, Cat: {s2.category_fit}, Merch: {s2.merchant_fit}, Dec: {s2.decision_quality}, Eng: {s2.engagement_compulsion}")
if s2.penalties:
    print(f"  Penalties: -{s2.penalties}")

time.sleep(4)

# Test 3: Salon Festive
cat3 = loader.categories['salons']
m3 = loader.merchants['m_003_studio11_salon_hyderabad']
trg3 = loader.triggers['trg_006_festival_diwali']
a3 = {
    'body': "Lakshmi ji, Diwali is approaching and festive queries for salons in Kapra are already surging. I've prepared a festive campaign draft featuring your active Hair Spa @ ₹499 and Haircut @ ₹99 packages to capture early bookings. Want me to schedule it on your Google profile? Reply YES.",
    'cta': 'binary',
    'send_as': 'vera'
}
s3 = scorer.score(a3, cat3, m3, trg3, None)
print(f"Test 3 (Salon Festival) Score: {s3.total}/50")
print(f"  Spec: {s3.specificity}, Cat: {s3.category_fit}, Merch: {s3.merchant_fit}, Dec: {s3.decision_quality}, Eng: {s3.engagement_compulsion}")
if s3.penalties:
    print(f"  Penalties: -{s3.penalties}")
