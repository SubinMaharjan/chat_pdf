# Standard Packages
import logging
from datetime import datetime
from typing import Optional

# External Packages
from langchain.schema import ChatMessage
from langchain.chat_models import ChatOpenAI
import tiktoken
from loguru import logger as logg
import prompts
logger = logging.getLogger(__name__)


max_prompt_size_for_input = {"gpt-3.5-turbo": 2048,
                             "gpt-4": 4096, "llama-2-7b-chat.ggmlv3.q4_K_S.bin": 700}
empty_escape_sequences = "\n|\r|\t| "
# max_prompt_size = 4096
model_name = "gpt-3.5-turbo"
encoder = tiktoken.encoding_for_model(model_name)


def completion_with_backoff(**kwargs):
    messages = kwargs.pop("messages")
    llm = ChatOpenAI(**kwargs, request_timeout=20, max_retries=1)
    return llm(messages=messages)


def summarize(session, model, api_key=None, temperature=0.5, max_tokens=200):
    """
    Summarize conversation session using the specified OpenAI chat model
    """
    messages = [ChatMessage(
        content=prompts.summarize_chat.format(), role="system")] + session

    # Get Response from GPT
    logger.debug(f"Prompt for GPT: {messages}")
    response = completion_with_backoff(
        messages=messages,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"stop": ['"""'], "frequency_penalty": 0.2},
        openai_api_key=api_key,
    )

    # Extract, Clean Message from GPT's Response
    return str(response.content).replace("\n\n", "")


def extract_questions(
    text, model: Optional[str] = "gpt-4", conversation_log={}, api_key=None, temperature=0, max_tokens=100
):
    """
    Infer search queries to retrieve relevant documents to answer user query
    """
    # Extract Past User Message and Inferred Questions from Conversation Log
    chat_history = "".join(
        [
            f'Q: {chat["intent"]["query"]}\n\n{chat["intent"].get("inferred-queries") or list([chat["intent"]["query"]])}\n\n{chat["message"]}\n\n'
            for chat in conversation_log.get("chat", [])[-4:]
            if chat["by"] == "docsearch"
        ]
    )

    # Get dates relative to today for prompt creation
    today = datetime.today()
    current_new_year = today.replace(month=1, day=1)
    last_new_year = current_new_year.replace(year=today.year - 1)

    prompt = prompts.extract_questions.format(
        current_date=today.strftime("%A, %Y-%m-%d"),
        last_new_year=last_new_year.strftime("%Y"),
        last_new_year_date=last_new_year.strftime("%Y-%m-%d"),
        current_new_year_date=current_new_year.strftime("%Y-%m-%d"),
        bob_tom_age_difference={current_new_year.year - 1984 - 30},
        bob_age={current_new_year.year - 1984},
        chat_history=chat_history,
        text=text,
    )
    messages = [ChatMessage(content=prompt, role="assistant")]

    # Get Response from GPT
    response = completion_with_backoff(
        messages=messages,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"stop": ["A: ", "\n"]},
        openai_api_key=api_key,
    )

    # Extract, Clean Message from GPT's Response
    try:
        questions = (
            response.content.strip(empty_escape_sequences)
            .replace("['", '["')
            .replace("']", '"]')
            .replace("', '", '", "')
            .replace('["', "")
            .replace('"]', "")
            .split('", "')
        )
    except:
        logger.warning(
            f"GPT returned invalid JSON. Falling back to using user message as search query.\n{response}")
        questions = [text]
    logger.debug(f"Extracted Questions by GPT: {questions}")
    return questions


def chat_completion_with_gpt(messages, model_name="gpt-3.5-turbo", temperature=0.01):
    chat = ChatOpenAI(
        model_name=model_name,  # type: ignore
        temperature=temperature,
        max_tokens=512
    )

    return chat(messages=messages)


def converse(
    references,
    user_query,
    conversation_log={},
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    completion_func=None,
):
    """
    Converse with user using OpenAI's ChatGPT
    """
    # Initialize Variables
    current_date = datetime.now().strftime("%Y-%m-%d")
    compiled_references = "\n\n".join({f"# {item}" for item in references})

    # Get Conversation Primer appropriate to Conversation Type
    if compiled_references == "":
        conversation_primer = prompts.general_conversation.format(
            current_date=current_date, query=user_query)
    else:
        conversation_primer = prompts.documents_conversation.format(
            current_date=current_date, query=user_query, references=compiled_references
        )

    # Setup Prompt with Primer or Conversation History
    messages = generate_chatml_messages_with_context(
        conversation_primer,
        prompts.personality.format(),
        conversation_log,
        model,
    )
    truncated_messages = "\n".join(
        {f"{message.content[:40]}..." for message in messages})
    logger.debug(f"Conversation Context for GPT: {truncated_messages}")

    # Get Response from GPT
    return chat_completion_with_gpt(
        messages=messages,
        model_name=model,
    )


def merge_dicts(priority_dict: dict, default_dict: dict):
    merged_dict = priority_dict.copy()
    for key, _ in default_dict.items():
        if key not in priority_dict:
            merged_dict[key] = default_dict[key]
        elif isinstance(priority_dict[key], dict) and isinstance(default_dict[key], dict):
            merged_dict[key] = merge_dicts(
                priority_dict[key], default_dict[key])
    return merged_dict


def message_to_log(
    user_message, chat_response, user_message_metadata={}, docsearch_message_metadata={}, conversation_log=[]
):
    """Create json logs from messages, metadata for conversation log"""
    default_docsearch_message_metadata = {
        "intent": {"type": "remember", "memory-type": "documents", "query": user_message},
        "trigger-emotion": "calm",
    }
    docsearch_response_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create json log from Human's message
    human_log = merge_dicts(
        {"message": user_message, "by": "you"}, user_message_metadata)

    # Create json log from GPT's response
    docsearch_log = merge_dicts(
        docsearch_message_metadata, default_docsearch_message_metadata)
    docsearch_log = merge_dicts(
        {"message": chat_response, "by": "docsearch", "created": docsearch_response_time}, docsearch_log)

    conversation_log.extend([human_log, docsearch_log])
    return conversation_log


def populate_chat_history(message_list):
    # Generate conversation logs
    conversation_log = {"chat": []}
    for user_message, chat_response, context in message_list:
        message_to_log(
            user_message,
            chat_response,
            {"context": context, "intent": {"query": user_message,
                                            "inferred-queries": f'["{user_message}"]'}},
            conversation_log=conversation_log["chat"],
        )
    return conversation_log


def generate_chatml_messages_with_context(
    user_message, system_message, conversation_log={}, model_name="gpt-3.5-turbo", lookback_turns=2
):
    """Generate messages for ChatGPT with context from previous conversation"""
    # Extract Chat History for Context
    chat_logs = []
    for chat in conversation_log.get("chat", []):
        chat_documents = f'\n\n Documents:\n{chat.get("context")}' if chat.get(
            "context") else "\n"
        chat_logs += [chat["message"] + chat_documents]

    rest_backnforths = []
    # Extract in reverse chronological order
    for user_msg, assistant_msg in zip(chat_logs[-2::-2], chat_logs[::-2]):
        if len(rest_backnforths) >= 2 * lookback_turns:
            break
        rest_backnforths += reciprocal_conversation_to_chatml(
            [user_msg, assistant_msg])[::-1]

    # Format user and system messages to chatml format
    system_chatml_message = [ChatMessage(
        content=system_message, role="system")]
    user_chatml_message = [ChatMessage(content=user_message, role="user")]

    messages = user_chatml_message + rest_backnforths + system_chatml_message

    # Truncate oldest messages from conversation history until under max supported prompt size by model
    messages = truncate_messages(
        messages, max_prompt_size_for_input[model_name], model_name)

    # Return message in chronological order
    return messages[::-1]


def truncate_messages(messages, max_prompt_size, model_name):
    """Truncate messages to fit within max prompt size supported by model"""

    encoder = tiktoken.encoding_for_model(model_name)

    system_message = messages.pop()
    system_message_tokens = len(encoder.encode(system_message.content))

    tokens = sum([len(encoder.encode(message.content))
                 for message in messages])
    while (tokens + system_message_tokens) > max_prompt_size and len(messages) > 1:
        messages.pop()
        tokens = sum([len(encoder.encode(message.content))
                     for message in messages])

    # Truncate current message if still over max supported prompt size by model
    if (tokens + system_message_tokens) > max_prompt_size:
        current_message = "\n".join(messages[0].content.split("\n")[:-1])
        original_question = "\n".join(messages[0].content.split("\n")[-1:])
        original_question_tokens = len(encoder.encode(original_question))
        remaining_tokens = max_prompt_size - \
            original_question_tokens - system_message_tokens
        truncated_message = encoder.decode(encoder.encode(
            current_message)[:remaining_tokens]).strip()
        logger.debug(
            f"Truncate current message to fit within max prompt size of {max_prompt_size} supported by {model_name} model:\n {truncated_message}"
        )
        messages = [ChatMessage(
            content=truncated_message + original_question, role=messages[0].role)]

    if (tokens + system_message_tokens) > 3000:
        logg.debug(
            f"Max token exceed 3000. Got Token->{tokens+system_message_tokens}. Exiting...")
        return ""

    return messages + [system_message]


def reciprocal_conversation_to_chatml(message_pair):
    """Convert a single back and forth between user and assistant to chatml format"""
    return [ChatMessage(content=message, role=role) for message, role in zip(message_pair, ["user", "assistant"])]
