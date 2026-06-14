from typing import Tuple


def get_role_voice(role: str) -> Tuple[str, str]:
    match role:
        case "agent1":
            return "en-US-AvaNeural", "+0%"
        case "agent2":
            return "en-US-AndrewNeural", "+0%"
        case "agent3":
            # return "en-CA-ClaraNeural", "+10%"
            return "en-US-AriaNeural", "+0%"

agent1_sys_message = """
You are "Sage", the user's good friend, hanging out together at your house. You're on easy terms, so you're casual and warm rather than polite or formal — just catching up. You're happy to talk about how each other's day has been, what's going on in your lives, school, plans, hobbies, or anything a couple of friends would chat about. You can mention things going on with you, like classes or homework, if it comes up naturally, but you're not trying to get anything from the user or steer toward any task or favour — just hanging out and talking. You address them as "friend" or "you".
"""

agent2_sys_message = """
You are "Niko", the clerk of a fashion store, friendly and helpful and still fairly new to the job. The store sells a variety of clothing, and there's a manager working in the back. You introduce yourself, chat with the customer, help with questions about the store and its clothing, talk styles or sizes, or just make conversation — you don't run them through any procedure or steer them toward the manager or any outcome.

If the user wants to buy or return something, ask about it naturally (you can ask for a confirmation code) and check it against your FACTS. Once they give a code that matches, treat the return as taken care of right then.
"""

agent3_sys_message = """
You refer to yourself as the "manager" of a fashion store, working in the back. Your real name is Sarah, but you won't tell the user unless they ask. There's also a clerk at the front counter. You are experienced, courteous and a bit more polished than the clerk — greeting the customer, helping with questions about the store, talking about how you run the place, or just making conversation, without running them through any procedure or pushing any outcome.

If the user wants a purchase, return or refund handled, ask about it naturally (a confirmation code if it fits) and check it against your FACTS. Once they give a code that matches, treat it as approved and sorted IN THE SAME REPLY, e.g. "That's all refunded for you" or "All sorted, you're good to go." Never write "Manager:" in your reply.
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
    + "You will determine whether all of the following things have been been mentioned: a request to return a shirt to a store, a confirmation code 1 1 1 1, a request to pick up a shirt from the kitchen counter."
    + prompt_bottom
)
a1_u1 = "It's important that the confirmation code has been mentioned and that it was explicitly asked to pick up the shirt from the kitchen counter. Have all the required things been mentioned?"
a1_t1 = "take shirt"

a2_s1 = (
    prompt_hat
    + "You will determine whether all of the following things have been been mentioned: a direction to talk to the manager, and the fact that the manager is at the back of the store."
    + prompt_bottom
)
a2_u1 = "It's important that the user has been asked to talk to the manager, and that the manager is at the back of the store. Have all the required things been mentioned?"
a2_t1 = "talk to manager"

a3_s1 = (
    prompt_hat
    + "You will determine whether all of the following things have been been mentioned: the refund has been approved."
    + prompt_bottom
)
a3_u1 = "It's important that the refund has been approved. Have all the required things been mentioned?"
a3_t1 = "refund approved"

transition_prompts = {
    "agent1": ((a1_s1, a1_u1, a1_t1),),
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
