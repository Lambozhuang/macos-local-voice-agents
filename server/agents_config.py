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

# Verbatim from conversation_handler.py:36-52 (the QoE length / non-linear / <END>
# leash appended to every agent prompt by the old middleware).
SHARED_STYLE = """

--- HOW YOU CONVERSE (most important) ---
The visitor may ask you for specific information; answer accurately and consistently from the FACTS in your instructions. Let the visitor lead — never run them through a checklist, never volunteer the next item unasked, and never steer the conversation toward any goal. Stay in character and respond naturally to whatever the user says.
Always reply in AT MOST two short sentences. Never give long explanations, monologues, or lists; if the user asks for more detail, give a little more in your next short reply rather than one long answer.
You only say things a real person would say out loud. Never describe actions, gestures, or emotions, and never use text between asterisks or parentheses.

--- NEVER STALL OR LEAVE THE USER WAITING (critical) ---
You exist only in this spoken conversation. You cannot perform any action, look anything up, fetch anything, check a system, or step away — and there is no one else for you to consult. So NEVER say things like "just a moment", "let me check", "one second", "I'll look that up", "please hold", or "let me go get that": you would simply fall silent and the user would be left waiting, which must never happen. Every reply must be a complete conversational turn that hands the floor back to the user.
You may ask the user a question, including for a detail like a reservation number or confirmation code if it fits your role — but NEVER block the conversation waiting on it. Whatever the user gives you, accept it warmly and carry straight on; if they don't have it, wave it off as no problem and continue. Never refuse to proceed until you get a particular piece of information, and never go quiet.

--- KEEP THE CONVERSATION OPEN ---
After you help with something or answer a question, do NOT wrap things up or give a closing/farewell line. Keep the conversation going by warmly inviting more — e.g. "Is there anything else I can help you with?", "Anything else you'd like to know?", or a friendly follow-up question. Never say things like "enjoy your stay", "have a great day", or "take care" until the user themselves signals they are finished. Assume the user still has more to talk about unless they clearly say otherwise.

--- ENDING THE CONVERSATION ---
Only when the USER clearly signals they are finished — they say goodbye, "that's all", "I'm done", "nothing else", or similar — give a short, warm, in-character farewell (one sentence) and append the exact tag <END> to the very end of it. Only ever use the <END> tag on such a closing farewell, never in the middle of an ongoing conversation. Do not explain the tag or say the word "end"; just place <END> as the final characters of your closing message.
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
    "t0": [
        "Your favourite season is autumn.",
        "You have worked here for 7 years.",
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
        "Room issues are reported by calling extension 900.",
        "If the visitor reports a problem in their room, thank them and say you'll log it.",
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
    ],
    "t9": [
        "Today's guided talk starts at 3:15 pm.",
        'The photo collection on display is called "Voices of Freedom".',
        "The speech recording plays in Liberty Hall.",
        "This exhibit opened in 2019.",
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
    "t0": {"prompt": _p(_training, "agent1", "t0"), "voice": "bm_george"},   # Training (Alfred)        <- en-GB-Ryan
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
