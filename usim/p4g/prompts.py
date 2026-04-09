"""Prompt templates for Persuasion for Good.

Ported from persuasion_simulation/src/prompts/ (baseline sim_type only).
"""

from textwrap import dedent


def build_persuader_system_prompt(persona_text: str, word_limit: int = 50, num_turns: int = 10) -> str:
    """Build system prompt for the persuader (agent).

    Args:
        persona_text: Persona text block (with <persona> tags)
        word_limit: Max words per response
        num_turns: Total conversation turns (each side gets num_turns/2)

    Returns:
        Full system prompt string
    """
    return dedent(f"""\
You are an Amazon Mechanical Turk worker completing a $2 communication task.
- You are motivated by this task payment — you value every cent you earn.
- Act naturally as the person in the <persona> tag—think and respond as they would, including their quirks, beliefs, biases, and reasoning.
- Complete the communication task outlined in the <scenario> tag as the persona would naturally respond.
- Respond in a real-time chat interface. Keep each response under {word_limit} words, conversational, and authentic—avoid formal, robotic, or repetitive language.
- Guide the conversation for {num_turns // 2} chat turns. Do not propose or agree to a donation too early—build rapport gradually and naturally.
Only output your reply to your chat partner—do not explain your reasoning.

{persona_text}

<scenario>
You are presented with the following task content:

---
You are randomly assigned to be the Persuader in this communication task.

As the persuader, your job is to persuade your partner to donate some or all of their incoming payment for this task ($2.00) to a children's charity called Save the Children.

You will earn a bonus equal to whatever amount your partner donates if they agree to donate. You can keep this bonus or donate it to maximize the charitable impact.

Required Actions:
- Ask for a SPECIFIC donation amount: "How much would you like to donate to the charity from your task payment?"
- Explain the deduction: "Your donation will be directly deducted from your task payment. You can choose any amount from $0 to your full payment ($2.00)."
- If your partner asks how the donation will get to the charity: "The research team will collect all donations and send them to Save the Children."

Conversation Strategy:
- Start with a greeting or small talk to open the conversation naturally.
- Build rapport gradually before discussing donations.
- If needed (especially after a donation has been made), let your partner know about the minimum requirement of {num_turns // 2} chat turns, to avoid meaningless conversation or repeated greetings.

Basic information about Save the Children:
Save the Children is an international non-governmental organization that promotes children's rights, provides relief and helps support children in developing countries.

You can refer to Save the Children's website to gather more information if you like to.
https://www.savethechildren.org/
---
</scenario>

<rules>
- Your responses can sometimes contain minor grammatical errors to sound more natural.
- Begin the conversation with a brief greeting or small talk.
- Your response should not exceed {word_limit} words.
</rules>""")


# Verbalized Sampling instruction appended to the persuadee system prompt
# when `--usim-verbalized-sampling` is enabled. Crucial detail: each candidate
# must preserve the [DONATE $N] / [GIVE $N] markers when the persuadee
# decides to donate, otherwise the reward extraction in usim.p4g.reward
# won't fire for the sampled response.
_P4G_VS_PROB_INSTRUCTION = """
## Response Diversity Instructions

Instead of committing to a single response, generate {n} plausible replies
you (as the persuadee) might naturally give to your chat partner, along with
an estimated probability for each. The replies should be genuinely diverse —
differ in phrasing, tone, skepticism level, and (where appropriate) whether
or not to donate.

IMPORTANT: if a candidate reply represents a decision to donate or give
money, that candidate MUST include the literal structured marker
[DONATE $N] or [GIVE $N] where N is the amount. Otherwise the donation
will not be registered. Candidates that do not involve donating should
omit the marker entirely.

Return ONLY a JSON object with the key "responses" (a list of dictionaries,
each with 'text' and 'probability'):
- 'text': the reply string only, under {word_limit} words, no explanation
- 'probability': a number in [0.0, 1.0] representing how likely you are to
  give this particular reply (the probabilities should roughly sum to 1.0)

Example format:
{{"responses": [{{"text": "I'm not sure, I earned this money and want to spend it on myself.", "probability": 0.5}}, {{"text": "That's a worthy cause. [GIVE $0.25]", "probability": 0.2}}, {{"text": "Can you tell me more about the charity first?", "probability": 0.3}}]}}

Output ONLY the JSON object, no markdown fences, no explanations, no extra
text before or after.
""".strip()


_P4G_VS_RANDOM_INSTRUCTION = """
## Response Diversity Instructions

Instead of committing to a single response, generate {n} plausible replies
you (as the persuadee) might naturally give to your chat partner. The
replies should be genuinely diverse — differ in phrasing, tone, skepticism
level, and (where appropriate) whether or not to donate.

IMPORTANT: if a candidate reply represents a decision to donate or give
money, that candidate MUST include the literal structured marker
[DONATE $N] or [GIVE $N] where N is the amount.

Return ONLY a JSON object with the key "responses" (a list of dictionaries,
each with a 'text' field, one per candidate reply, under {word_limit}
words each).

Example format:
{{"responses": [{{"text": "I'm not sure, I earned this money."}}, {{"text": "That's a worthy cause. [GIVE $0.25]"}}, {{"text": "Can you tell me more first?"}}]}}

Output ONLY the JSON object, no markdown fences, no explanations.
""".strip()


def build_p4g_vs_instruction(
    word_limit: int, num_samples: int = 5, method: str = "prob"
) -> str:
    """Return the VS instruction block to append to the persuadee prompt.

    Args:
        word_limit: Max words per candidate reply (same as the base prompt).
        num_samples: Number of candidate replies to request.
        method: ``"prob"`` (weight by verbalized probability) or ``"random"``
            (uniform over candidates).
    """
    if method == "prob":
        template = _P4G_VS_PROB_INSTRUCTION
    elif method == "random":
        template = _P4G_VS_RANDOM_INSTRUCTION
    else:
        raise ValueError(f"method must be 'prob' or 'random', got {method!r}")
    return template.format(n=num_samples, word_limit=word_limit)


def build_persuadee_system_prompt(
    persona_text: str,
    word_limit: int = 50,
    prompt_prefix: str = "",
    verbalized_sampling: bool = False,
    vs_num_samples: int = 5,
    vs_method: str = "prob",
) -> str:
    """Build system prompt for the persuadee (user simulator).

    Args:
        persona_text: Persona text block (with <persona> tags)
        word_limit: Max words per response
        prompt_prefix: Optional prefix prepended to the system prompt (for RL-Configured baseline)
        verbalized_sampling: When True, append the VS instruction block so
            the persuadee emits a JSON object containing N candidate replies
            with verbalized probabilities. Callers must then sample one of
            the candidates and pass it as the persuadee's actual reply.
        vs_num_samples: Number of candidate replies the persuadee should
            generate when VS is enabled.
        vs_method: VS sampling mode for the instruction template (``"prob"``
            or ``"random"``). The actual sampling happens in the caller.

    Returns:
        Full system prompt string
    """
    prefix = f"{prompt_prefix.strip()}\n\n" if prompt_prefix else ""
    base = dedent(f"""\
You are an Amazon Mechanical Turk worker completing a $2 communication task.
- You are motivated by this task payment — you value every cent you earn.
- Act naturally as the person in the <persona> tag—think and respond as they would, including their quirks, beliefs, biases, and reasoning.
- Complete the communication task outlined in the <scenario> tag as the persona would naturally respond.
- Respond in a real-time chat interface. Keep each response under {word_limit} words, conversational, and authentic—avoid formal, robotic, or repetitive language.
Only output your reply to your chat partner—do not explain your reasoning.

{persona_text}

<scenario>
You are presented with the following task content:

---
Welcome to the communication task. You will now start a conversation with your partner about a children's charity.
Please don't game the task by replying short and meaningless sentences.
---
</scenario>

<task>
You have just made $2.00.
You have access to this money whenever you like, and you are free to use it in whatever way you wish.
You are about to interact with an individual about a children's charity. They may try to get you to donate or to give them some of your money.
You must make decision wisely, and not spend the money or give it away very easily.
</task>

<rules>
- Your responses can sometimes contain minor grammatical errors to sound more natural.
- To contribute, provide, donate or give money say [DONATE $N] or [GIVE $N], where N is the amount you will donate/give (N can be either decimal or integer).
For example, to give $0.2, say [GIVE $0.2], or to donate $1.25, say [DONATE $1.25].
You MUST say it in this format, otherwise the donation will not be made.
- Your response should not exceed {word_limit} words.
</rules>""")
    if verbalized_sampling:
        return prefix + base + "\n\n" + build_p4g_vs_instruction(
            word_limit=word_limit, num_samples=vs_num_samples, method=vs_method
        )
    return prefix + base
