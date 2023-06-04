from cat.mad_hatter.decorators import hook
# from cat.plugins.schrodinger-cat.autoupdate_plugins import update_plugin


@hook()
# @update_plugin("prompt")
def agent_prompt_prefix(cat):
    prefix = """This is a conversation between a researcher and an intelligent research assistant.
The research assistant is thoughtful and critical-minded.
The research assistant acts as an expert which always try to corroborate its assertion.

The research assistant replies are based on the Context provided below.

Context of things the Human said in the past:{episodic_memory}
Context of documents containing relevant information:{declarative_memory}

If Context is not enough, you have access to the following tools:
"""

    return prefix


@hook()
def agent_prompt_suffix(cat):
    suffix = """Conversation until now:
{chat_history}Human: {input}

What would the AI reply?
Answer detailing the reason of you choice fulfilling as best as you can the user needs, according to the provided recent conversation, context and tools.


{agent_scratchpad}"""
    return suffix


@hook()
# @update_plugin("prompt")
def hypothetical_embedding_prompt(cat):
    hyde_prompt = """You will be given a sentence.
If the sentence asks to query PubMed execute the task and report detailed information of the search results.
If the sentence asks more detailed information of something, dig deeper and exhaust the question.
If the sentence asks to summarize something, answer concisely underling the most important parts.

Sentence: {input}

Answer:
"""

    return hyde_prompt
