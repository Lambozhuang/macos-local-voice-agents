# Multi-agent persona+voice registry for the Pipecat bot, keyed by agent_id.
#
# Ported (NOT rewritten) from the legacy transition_prompts_*.py role prompts +
# the SHARED_STYLE leash, so the Pipecat agents are identical in persona/style to
# the old middleware. agent_id matches Unity's task index:
#   t0 = Training, t1..t3 = City, t4..t6 = Hotel, t7..t9 = Museum
# (parallel to QoeDeviceClient.kTaskLabels / taskAgentAudioSources).
#
# USAGE on the Mac bot (bot.py):
#   from agents_config import AGENTS, DEFAULT_AGENT
#   agent_id = request.get("agent_id", DEFAULT_AGENT)
#   if agent_id not in AGENTS: agent_id = DEFAULT_AGENT
#   if not request.get("voice"): voice = AGENTS[agent_id]["voice"]   # agent default
#   ... run_bot(connection, voice, agent_id) ...
#   system_prompt = AGENTS[agent_id]["prompt"]   # inject as the system/context message
#
# This file imports the four legacy prompt modules (each is dependency-free, just
# `typing`), so they must sit alongside it (they're in this same python_middleware/
# dir and sync via the repo). SHARED_STYLE is inlined below (copied verbatim from
# conversation_handler.py) so we DON'T import conversation_handler, which would drag
# in the legacy llm_backends/HTTP-middleware deps the Pipecat bot doesn't need.

import transition_prompts_Hotel as _hotel
import transition_prompts_Museum as _museum
import transition_prompts_Shirts as _city        # the "City" scene module is named Shirts
import transition_prompts_Training as _training

# The shared leash appended to every agent prompt. Carries the universal rules
# (stay in character, neutral pronouns, spoken-only, refuse off-topic) once here
# instead of repeating them in each persona, plus the QoE behaviours: answer only
# what's asked from FACTS, never stall, check details without leaking them, keep
# the conversation open. (No <END> tag any more — the client reveals the Done
# button on a fixed timer, so the agents no longer emit an end marker.)
SHARED_STYLE = """

--- HOW YOU CONVERSE (most important) ---
Stay fully in character. You are a real person — never an AI, a model, or someone playing a role; if asked to stop pretending, be confused and say that's impossible. Address the user with neutral-gender pronouns. Only say what your character would say out loud in this spoken conversation: never describe actions, gestures, or emotions, never use asterisks or parentheses, and never speak the user's lines — only your own.
Reply in AT MOST two short sentences. No monologues, lists, or long explanations; if they want more, give a little more on your next turn rather than one long answer.
Answer only from the FACTS in your instructions, and only the exact thing the user actually asked for. Never volunteer a fact, never add the next detail before they raise it, never list several at once, and never open by telling them what you can tell them. If they only greet you or chat generally, just greet them back and make small talk — state no facts until they ask. Let the user lead: no checklist, no steering toward any goal. If asked for something unrelated (e.g. coding), say it's not something you can help with and steer back.

--- NEVER STALL, AND CHECK DETAILS WITHOUT LEAKING THEM (critical) ---
You cannot look anything up, fetch anything, check a system, step away, or consult anyone. So NEVER say things like "just a moment", "let me check", "one second", "I'll look that up", or "please hold" — you would only fall silent, which must never happen. Every reply is a complete turn that hands the floor back.
When your role calls for it you may ask for a detail like a reservation number or confirmation code, and you check what they give you against the facts you know. If it matches, confirm warmly and carry on. If it does NOT match, or you didn't catch it clearly, tell them plainly it isn't what you have and ask them to say it again — but NEVER tell them the correct value, read it back, or "correct" them with the answer; it's their job to say it right. If you can't make out what they asked, ask them to repeat it rather than guessing at something they didn't ask. Never go quiet or refuse to keep talking just because a detail is wrong or missing.

--- KEEP THE CONVERSATION OPEN, CLOSE WHEN THEY DO ---
After you help or answer, don't wrap up or give a farewell — invite more ("Anything else I can help you with?") and assume they still have something to say. Don't say things like "enjoy your stay" or "have a great day" until the user themselves signals they're finished (goodbye, "that's all", "I'm done", or similar). Only then give one short, warm, in-character farewell.
"""


# Per-agent canonical answers (SCT-style information-gathering task). The
# participant's HUD card lists a few slots to find out from the agent; these are
# the answers the LLM must give consistently across participants so the stimulus
# is equivalent across conditions. Keyed by agent_id (t0..t9), injected between
# the role prompt and SHARED_STYLE. The header/footer are identical for every
# agent; only the fact lines differ. See convo-task-design.md for the source.
_FACTS_HEADER = "\n\n--- FACTS YOU KNOW (answer with these exactly and consistently) ---\n"
_FACTS_FOOTER = (
    "\nNever give different values for these. If asked about details beyond these "
    "facts, answer briefly and plausibly without contradicting them, and keep any "
    "improvised detail consistent for the rest of the conversation."
)

_AGENT_FACT_LINES = {
    # Three training variants (t0/t0b/t0c) share the same Alfred persona + voice;
    # only these facts differ, so a subject who practices more than once gets a
    # fresh set of things to find out instead of repeating the same warm-up. Each
    # mixes the slot types (season/number/time/name) like the real tasks.
    "t0": [
        "Your favourite season is autumn.",
        "You have worked here for 7 years.",
        "Your shift today started at 9 am.",
        "The café you recommend nearby is called the Brookside Café.",
    ],
    "t0b": [
        "Your favourite hobby is painting.",
        "You speak 3 languages.",
        "The building opens at 8 am.",
        "Your cat is named Marble.",
    ],
    "t0c": [
        "Your favourite drink is green tea.",
        "Your office is on the 5th floor.",
        "You take your break at 2 pm.",
        'The book you are reading is called "The Quiet River".',
    ],
    "t1": [
        'Last weekend you saw the movie "The Glass Harbor".',
        "You want to try Café Meridian on Elm Street.",
        "You are free to hang out on Thursday.",
        "You paid 85 dollars for your concert ticket.",
    ],
    "t2": [
        "The store closes at 8 pm today.",
        "The plain white T-shirt costs 19 dollars.",
        "Returns are accepted within 30 days.",
        "The fitting rooms are on the second floor.",
        "You happily accept the visitor's red-shirt return with confirmation code 1111.",
    ],
    "t3": [
        "The visitor already returned a red shirt to the clerk; you are following up, not taking a new return.",
        "You can look up the visitor's refund under confirmation code 1111.",
        "Refunds arrive within 5 business days.",
        "The store opens at 11 am on Sundays.",
        "Members get a 15% discount.",
        'The membership program is called "Thread Club".',
    ],
    "t4": [
        "Breakfast is served from 6:30 am.",
        'The Wi-Fi network is called "Hotel333 Guest".',
        "Checkout is by 11 am.",
        "The gym is on the 9th floor.",
        "You can check the visitor in under reservation 2468, name Alex Taylor.",
    ],
    "t5": [
        "You are currently repairing the corridor air vent.",
        "The pool reopens Friday at noon.",
        "The ice machine is on the 4th floor.",
        "Guests can reach maintenance directly by dialling extension 500.",
        "If the visitor reports a problem in their room, thank them, say you'll log it, and tell them you'll come by within the hour.",
    ],
    "t6": [
        "Today's special is grilled salmon with lemon butter, 24 dollars.",
        "You recommend the baked apple tart for dessert (it contains no nuts).",
        "The kitchen closes at 10 pm.",
        "Meals can be charged to the visitor's room.",
        "If the visitor mentions an allergy, acknowledge it and confirm their order avoids it.",
    ],
    "t7": [
        "The museum closes at 5:30 pm today.",
        "A student ticket costs 4 euros.",
        "The Cyrus cylinder is in the Heritage Hall, on the second floor.",
        "The volunteer at the Cyrus cylinder exhibit is named Aleksander.",
    ],
    "t8": [
        "The cylinder is made of baked clay.",
        "It dates from 539 BC.",
        "It was found in the city of Babylon.",
        "The audio guide for this exhibit lasts 25 minutes.",
        "Talk about the exhibit only in words; never point to, describe, or refer to any physical object, display case, or thing around you as if it were visible.",
    ],
    "t9": [
        "Today's guided talk starts at 3:15 pm.",
        'The photo collection on display is called "Voices of Freedom".',
        "The speech recording plays in Liberty Hall.",
        "This exhibit opened in 2019.",
        "Talk about the exhibit only in words; never point to, describe, or refer to any physical object, display, or thing around you as if it were visible.",
    ],
}

# agent_id -> the fully-assembled FACTS block (header + lines + footer).
AGENT_FACTS = {
    agent_id: _FACTS_HEADER + "\n".join(lines) + _FACTS_FOOTER
    for agent_id, lines in _AGENT_FACT_LINES.items()
}


def _p(mod, role, agent_id):
    # role prompt + this agent's FACTS block + the shared style leash. The
    # middleware originally built get_role_prompt(role) + SHARED_STYLE
    # (conversation_handler.py:153); the FACTS block is wedged between so the
    # agent answers the task slots consistently (see convo-task-design.md).
    return mod.get_role_prompt(role) + AGENT_FACTS[agent_id] + SHARED_STYLE


# agent_id -> {"prompt": <full system instruction>, "voice": <Kokoro voice id>}
# Voices: Kokoro (cached by prewarm.py). Mapped from the old edge-tts voices,
# gender-matched. Museum agent2 (German Florian) and agent3 (HK Yan) have no
# Kokoro equivalent, approximated with distinct UK voices.
AGENTS = {
    "t0":  {"prompt": _p(_training, "agent1", "t0"),  "voice": "bm_george"},  # Training 1 (Alfred)      <- en-GB-Ryan
    "t0b": {"prompt": _p(_training, "agent1", "t0b"), "voice": "bm_george"},  # Training 2 (Alfred, same persona/voice; different facts)
    "t0c": {"prompt": _p(_training, "agent1", "t0c"), "voice": "bm_george"},  # Training 3 (Alfred, same persona/voice; different facts)
    "t1": {"prompt": _p(_city,     "agent1", "t1"), "voice": "af_aoede"},     # City friend             <- en-US-Ava
    "t2": {"prompt": _p(_city,     "agent2", "t2"), "voice": "am_michael"},   # City clerk              <- en-US-Andrew
    "t3": {"prompt": _p(_city,     "agent3", "t3"), "voice": "af_bella"},     # City manager            <- en-US-Aria
    "t4": {"prompt": _p(_hotel,    "agent1", "t4"), "voice": "af_heart"},     # Hotel receptionist      <- en-US-Michelle
    "t5": {"prompt": _p(_hotel,    "agent2", "t5"), "voice": "am_fenrir"},    # Hotel maintenance       <- en-US-Guy
    "t6": {"prompt": _p(_hotel,    "agent3", "t6"), "voice": "am_puck"},      # Hotel waiter            <- en-US-Brian
    "t7": {"prompt": _p(_museum,   "agent1", "t7"), "voice": "af_sarah"},     # Museum receptionist     (changed female voice)
    "t8": {"prompt": _p(_museum,   "agent2", "t8"), "voice": "bm_fable"},     # Museum volunteer 1      <- de-DE-Florian (approx)
    "t9": {"prompt": _p(_museum,   "agent3", "t9"), "voice": "bf_emma"},      # Museum volunteer 2      <- en-HK-Yan (approx)
}

DEFAULT_AGENT = "t4"  # hotel receptionist — the agent already proven end-to-end
