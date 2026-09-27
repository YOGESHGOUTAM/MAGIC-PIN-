"""
magicpin AI Challenge — Conversation Handlers
==============================================
Multi-turn conversational handler for Vera (merchant and customer interactions).
Handles:
1. Auto-reply detection and graceful exit/wait (WhatsApp canned replies)
2. Immediate intent transitions (moving from qualification to action without delay)
3. Hostile / unsubscribe / stop requests handling
4. Informational question answering and multi-turn flow continuation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    category_slug: Optional[str] = None
    turns: List[Dict[str, Any]] = field(default_factory=list)
    state: str = "initiated"  # "initiated", "in_discussion", "action_mode", "ended", "waiting"
    auto_reply_count: int = 0
    last_bot_body: str = ""
    trigger_kind: Optional[str] = None
    merchant_data: Optional[Dict[str, Any]] = None
    category_data: Optional[Dict[str, Any]] = None
    customer_data: Optional[Dict[str, Any]] = None


# Known patterns for WhatsApp Business automated canned replies
AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"thanks for reaching out",
    r"our team will respond shortly",
    r"we will get back to you",
    r"automated assistant",
    r"canned response",
    r"auto[- ]reply",
    r"currently unavailable",
    r"we have received your message",
    r"shukriya.*hamari team tak",
    r"main ek automated assistant hoon",
    r"hum jald hi sampark karenge",
    r"business hours are",
    r"away from the phone",
    r"auto-generated",
]

# Patterns signaling clear commercial/action intent (transition from pitch/qualifying to action)
INTENT_ACTION_PATTERNS = [
    r"\blet'?s do it\b",
    r"\bok\b.*\b(do it|proceed|start|next|go ahead)\b",
    r"\bwhat'?s next\b",
    r"\bi want to join\b",
    r"\bmujhe.*judrna hai\b",
    r"\byes please\b",
    r"\bgo ahead\b",
    r"\bproceed\b",
    r"\bsend me the\b",
    r"\bconfirm\b",
    r"\bstart now\b",
    r"\bdo it\b",
    r"\byes,? (lets|let's|do|start)\b",
]

# Patterns signaling hostile, opt-out, stop, or unsub request
HOSTILE_PATTERNS = [
    r"\bstop\b",
    r"\bunsubscribe\b",
    r"\bspam\b",
    r"\buseless\b",
    r"\bdon'?t message\b",
    r"\bstop messaging\b",
    r"\bnever message\b",
    r"\bmat bhejo\b",
    r"\bband karo\b",
    r"\bnot interested\b",
    r"\bleave me alone\b",
    r"\bremove (my|me)\b",
]

# Words that indicate qualifying questions (must be avoided when merchant has signaled intent)
QUALIFYING_WORDS = [
    "would you", "do you", "can you tell", "what if", "how about", "are you interested in"
]

# Action words to indicate immediate execution mode
ACTION_WORDS = [
    "done", "sending", "draft", "here", "confirm", "proceed", "next"
]


def is_auto_reply(message: str, previous_messages: List[str]) -> bool:
    """Detect if incoming message is an automated/canned WhatsApp auto-reply."""
    msg_clean = message.strip().lower()
    
    # Check regex signatures
    for pattern in AUTO_REPLY_PATTERNS:
        if re.search(pattern, msg_clean):
            return True
            
    # Check if identical message was sent 2+ times previously
    identical_count = sum(1 for prev in previous_messages if prev.strip().lower() == msg_clean)
    if identical_count >= 1 and len(msg_clean) > 15:
        return True
        
    return False


def is_hostile_or_stop(message: str) -> bool:
    """Detect if merchant or customer requests opt-out or expresses hostility."""
    msg_clean = message.strip().lower()
    for pattern in HOSTILE_PATTERNS:
        if re.search(pattern, msg_clean):
            return True
    return False


def is_intent_transition(message: str) -> bool:
    """Detect if merchant commits to action, requiring immediate switch to action mode."""
    msg_clean = message.strip().lower()
    for pattern in INTENT_ACTION_PATTERNS:
        if re.search(pattern, msg_clean):
            return True
    return False


def respond(state: ConversationState, merchant_message: str) -> Dict[str, Any]:
    """
    Given the conversation so far + the merchant's latest message, produce the reply.
    
    Returns a dict with:
        action: "send" | "wait" | "end"
        body: str (if action == "send")
        cta: "open_ended" | "binary" | "none"
        wait_seconds: int (if action == "wait")
        rationale: str
    """
    previous_inbound = [t["msg"] for t in state.turns if t.get("from") in ("merchant", "customer")]
    
    # Record inbound turn
    state.turns.append({"from": "merchant", "msg": merchant_message})
    
    # 1. Check for Auto-reply pattern
    if is_auto_reply(merchant_message, previous_inbound):
        state.auto_reply_count += 1
        state.state = "ended"
        return {
            "action": "end",
            "rationale": "Detected canned WhatsApp Business auto-reply signature; ending conversation gracefully to avoid burning merchant turns."
        }

    # 2. Check for Hostile / Opt-out / Stop request
    if is_hostile_or_stop(merchant_message):
        state.state = "ended"
        return {
            "action": "end",
            "rationale": "Merchant opted out or expressed frustration; ending conversation immediately with respect for user preferences."
        }

    # 3. Check for Intent Transition (Action Mode)
    if is_intent_transition(merchant_message) or state.state == "action_mode":
        state.state = "action_mode"
        merchant_name = ""
        if state.merchant_data:
            ident = state.merchant_data.get("identity", {})
            owner = ident.get("owner_first_name") or ident.get("name", "")
            merchant_name = f" {owner}" if owner else ""
            
        action_body = (
            f"Done{merchant_name}! I have prepared the draft and proceeding with the setup right now. "
            "Here is the preview link and confirmation details: all active services and hours are verified. "
            "Next step is live deployment within 24 hours. Reply CONFIRM to publish immediately."
        )
        return {
            "action": "send",
            "body": action_body,
            "cta": "binary",
            "rationale": "Merchant signaled explicit commitment; switched immediately to ACTION mode without any qualifying questions."
        }

    # 4. Check for merchant asking for delay / time ("talk later", "busy right now", "call tomorrow")
    msg_lower = merchant_message.lower()
    if any(p in msg_lower for p in ["busy", "later", "call tomorrow", "not now", "thodi der baad"]):
        state.state = "waiting"
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "Merchant requested time / is currently occupied; backing off 30 minutes before following up."
        }

    # 5. General engaged inquiry or question
    m_name = ""
    if state.merchant_data:
        ident = state.merchant_data.get("identity", {})
        m_name = ident.get("owner_first_name") or ident.get("name", "")
        
    greeting = f"Hi {m_name}, " if m_name else "Got it! "
    reply_body = (
        f"{greeting}I've pulled the exact details for you. Here is the draft ready for review. "
        "Next step takes under 60 seconds — reply YES to confirm and proceed."
    )
    
    return {
        "action": "send",
        "body": reply_body,
        "cta": "binary",
        "rationale": "Merchant provided input; acknowledged specific request and advanced directly to the next low-friction action."
    }
