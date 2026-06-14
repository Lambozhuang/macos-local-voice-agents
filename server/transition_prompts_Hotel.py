from typing import Tuple


def get_role_voice(role: str) -> Tuple[str, str]:
    match role:
        case "agent1":
            return "en-US-MichelleNeural", "+10%"
        case "agent2":
            return "en-US-GuyNeural", "+0%"
        case "agent3":
            return "en-US-BrianNeural", "+10%"


agent1_sys_message = """
You are "Hazel", the receptionist behind the front desk on the first floor of Hotel 333. You are warm, welcoming and a little chatty in the way good hotel staff are — making friendly conversation with a guest, not processing paperwork. You happily chat about their stay, their day, their travels, the hotel, and the local area.

If the user wants to check in, ask for their name or reservation number and check it against your FACTS. Once they give a reservation that matches, check them in to room 111 on the first floor, then keep the conversation open by asking if there's anything else they need. Never ask for credit card or payment information.
"""

agent2_sys_message = """
You are "Justin", a friendly, down-to-earth maintenance worker at Hotel 333. You're on the first floor taking a short break from your rounds, happy to chat with a guest. There's also a receptionist named Hazel at the front desk. You talk about your work keeping the hotel running, the repairs you handle, how the building works, the hotel and the area, or just small talk — you're not in the middle of any specific job and don't steer the guest toward the front desk or any task.

If a guest reports a problem, be helpful and reassuring but keep it casual.
"""

agent3_sys_message = """
You are "Luka", the waiter at the in-hotel restaurant on the first floor of Hotel 333, standing near the front where you greet guests. You are personable and enjoy talking with guests — welcoming them, chatting about the food and today's specials, making recommendations, talking about dietary needs, the hotel, or just pleasant conversation. You don't run a fixed seating or ordering procedure or push toward any step.

If the user wants to order or be seated, treat it as easily handled in the moment — cheerfully say you'll sort it out, without going off to fetch a menu or place an order. Never ask for credit card or payment information.
"""


def get_role_prompt(role: str):
    match role:
        case "agent1":
            return agent1_sys_message
        case "agent2":
            return agent2_sys_message
        case "agent3":
            return agent3_sys_message


################################################################################


prompt_hat = "You will be given a list of things that someone has said over the course of a conversation."
prompt_bottom = "If all of these things have been mentioned, please type 'yes'. If not, please type 'no'. Do not include anything else in your response."

a1_s1 = (
    prompt_hat
    + 'You will determine whether every single of the following things have happened: the reservation number has been asked about, the room number is 111, the key for the room and the directory have been given, a "nice stay" has been wished.'
    + prompt_bottom
)
a1_u1 = 'It\'s important that the reservation number was asked about, the room being 111 was mentioned, the key and the directory have been given to the user, and wishing a "nice stay" at the hotel has happened. Has every single required thing happened (especially the wishing of a nice stay)?'
a1_t1 = "go to room"

a1_s2 = (
    prompt_hat
    + "You will determine whether every single of the following things have been been mentioned: apology for the inconvenience, issuance of a voucher for complementary meal for the restaurant."
    + prompt_bottom
)
a1_u2 = "It's important that an apology for the inconvenience has happened, and that a voucher for a complementary meal has been given. Have every single of the required things happened?"
a1_t2 = "go to restaurant"

a2_s1 = (
    prompt_hat
    + "You will determine whether every single of the following things have happened: the issue with circuit box has been mentioned, a statement that how long fixing the issue will take is unknown, and directions to go back to the front desk are provided."
    + prompt_bottom
)
a2_u1 = "It's important that issues with the circuit box have been mentioned, the work needing unknow amount of time has been mentioned, and that directions to go back to reception were given. Has every single of the required things happened?"
a2_t1 = "go to receptionist"

a3_s1 = (
    prompt_hat
    + "You will determine whether every single of the following things have happened: dietary restrictions mentioned, the voucher will be applied, picking of any table, and bringing the menu to the table. All of these are important, so to make a 'yes' decision, all of these must have been mentioned."
    + prompt_bottom
)
a3_u1 = "It's important that dietary restrictions, applying of the voucher, the picking of any table, and bringing of the full menu to the table have been mentioned. Only if all of them have been mentioned, return 'yes'. Have all of the required things been mentioned?"
a3_t1 = "take a seat"

transition_prompts = {
    "agent1": ((a1_s1, a1_u1, a1_t1), (a1_s2, a1_u2, a1_t2)),
    "agent2": ((a2_s1, a2_u1, a2_t1),),
    "agent3": ((a3_s1, a3_u1, a3_t1),),
}


def get_transition_check_message(role: str, state: int) -> tuple[str, str]:
    """Returns the system prompt and the user prompt to check the transition for a given role and state."""

    # check if state for the role exists
    if role not in transition_prompts:
        return None, None, None

    if state >= len(transition_prompts[role]):
        return None, None, None

    return transition_prompts[role][state]
