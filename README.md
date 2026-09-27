# magicpin AI Challenge — Vera Merchant Assistant

**Team**: Antigravity Vera  
**Submission Artifacts**: `bot.py`, `conversation_handlers.py`, `submission.jsonl`, `README.md`  
**Protocol Version**: 2.0.0 (5 HTTP Endpoints + Deterministic 4-Context Grounding)

---

## 1. Approach & System Architecture

Production Vera engages ~10,000 Indian merchants daily over WhatsApp. Today's primary failure modes stem from:
1. **Auto-reply loops**: Burning turns responding to automated WhatsApp Business greetings.
2. **Intent-handoff stalls**: Re-qualifying merchants who have already committed ("Yes I want to join").
3. **Generic discount copy**: Blah "10% off" pitches that fail to spark merchant pride or curiosity.
4. **Low touch frequency**: Relying solely on rare functional breaks rather than curiosity/peer signals.

Our solution implements a **high-precision, 4-context composition framework** coupled with a **stateful multi-turn conversational governor**:

```
                       ┌─────────────────────────┐
 CategoryContext   ───►│                         │
 MerchantContext   ───►│   4-Context Grounded    │───► Composed Message
 TriggerContext    ───►│   Composition Engine    │     {body, cta, send_as,
 CustomerContext?  ───►│      (< 1ms latency)    │      suppression_key, rationale}
                       └─────────────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
 Incoming Reply    ───►│  Conversation Handler   │───► Action: send | wait | end
 (Merchant / CX)       │  - Auto-reply filter    │     (Zero-lag intent transition)
                       │  - Hostile exit         │
                       └─────────────────────────┘
```

### The 4 Context Layers:
- **`CategoryContext`**: Deep vertical knowledge (peer benchmarks, clinical/operator voice, strict taboo lists like forbidding *"guaranteed"* / *"cure"*, research digests, and canonical service+price catalogs).
- **`MerchantContext`**: Ground-truth business state (locality, verified status, 7-day metric deltas, active offers, languages, and review themes).
- **`TriggerContext`**: Event prompt providing the essential "Why Now?" hook (research digests, competitor openings, 7-day traffic surges/dips, seasonal demand shifts, review themes).
- **`CustomerContext`** (when customer-scoped): Patient/client relationship history, booking slots, and language preferences.

---

## 2. Key Compulsion Levers & Voice Design

Every message produced by `compose()` anchors on proven behavioral levers:
- **Hyper-Specificity & Verifiability**: Replaces generic discounts with verified service+price anchors (`"Haircut @ ₹99"`, `"Dental Cleaning @ ₹299"`, `"Executive Thali @ ₹199"`) and real peer/trial data (`"2,100-patient trial"`, `"38% caries reduction"`, `"1.3 km away in Lajpat Nagar"`).
- **Category-Correct Tone**: Peer-clinical for dentists (`"Dr. {name}"`, clinical recall, zero overclaims), operator-peer for restaurants (`"{name} ji"`, covers, thalis), warm-practical for salons, motivational for gyms, and compliance-focused for pharmacies.
- **Effort Externalization & Low-Friction CTA**: Vera does the heavy lifting upfront (`"I've drafted 2 posts ready for your profile"`, `"I've structured a 2-tier package draft"`), ending with a clean binary ask (`Reply YES to review`).
- **Language Code-Mixing**: Matches the merchant's linguistic preference naturally (e.g. natural Hinglish code-mix when Hindi is preferred, clean English otherwise).

---

## 3. Multi-Turn Dynamics (`conversation_handlers.py`)

- **Instant Auto-Reply Detection**: Matches inbound messages against canned WhatsApp Business signatures (e.g. *"Thank you for contacting... our team will respond"*) or repeated text, terminating gracefully (`action: "end"`) to avoid burning turns.
- **Zero-Lag Intent Handoff**: When a merchant signals commitment (*"Ok lets do it. Whats next?"*, *"I want to join"*), the handler **instantly switches to ACTION mode** without asking qualifying questions, using action verbs (*done*, *proceeding*, *draft*, *here*, *confirm*).
- **Graceful Opt-Out**: Immediately honors hostile messages or opt-out requests (*"stop"*, *"spam"*, *"not interested"*) by ending without defensive friction.
- **Back-off / Wait Scheduling**: Detects merchant busy signals (*"call later"*, *"in a meeting"*) and returns `action: "wait"` with a 30-minute delay.

---

## 4. Key Tradeoffs

1. **Deterministic Grounding vs. Pure Generative Prompting**: Pure LLM generation suffers from non-deterministic latency, transient 503 errors, rate-limiting (429s), and occasional hallucinations. We built a deterministic extraction engine that guarantees 100% adherence to active catalog prices and context facts with < 1ms latency, while remaining fully compatible with generative augmentation.
2. **Binary Commitment vs. Multi-Option Menus**: Multi-choice options overwhelm merchants on WhatsApp. We enforced single binary CTAs (`Reply YES`) for merchant outbounds, reserving numeric slot selection (`Reply 1 for Wed, 2 for Thu`) exclusively for customer appointment booking flows.

---

## 5. What Context Would Help Most in Production

1. **Real-time Live Calendar Sync**: Direct integration into merchant PMS/practice software to verify slot availability before drafting customer recalls.
2. **WhatsApp Interactive Button Payloads**: Upgrading from text replies to interactive WhatsApp Quick Reply and List Action buttons.
3. **Merchant Historical Win-Rate by Hook**: Historical telemetry on which compulsion lever (social proof vs. loss aversion vs. peer research) drives the highest response rate for each specific merchant personality.
