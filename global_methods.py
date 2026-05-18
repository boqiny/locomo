import openai
import numpy as np
import json
import time
import sys
import os
# Harbor-parity: anthropic + google.generativeai are lazy-imported inside the
# functions that use them so the gpt-5-mini path doesn't require them.


def get_openai_embedding(texts, model="text-embedding-ada-002"):
   texts = [text.replace("\n", " ") for text in texts]
   return np.array([openai.Embedding.create(input = texts, model=model)['data'][i]['embedding'] for i in range(len(texts))])

def set_anthropic_key():
    pass

def set_gemini_key():
    import google.generativeai as genai
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def set_openai_key():
    openai.api_key = os.environ['OPENAI_API_KEY']


def run_json_trials(query, num_gen=1, num_tokens_request=1000, 
                model='davinci', use_16k=False, temperature=1.0, wait_time=1, examples=None, input=None):

    run_loop = True
    counter = 0
    while run_loop:
        try:
            if examples is not None and input is not None:
                output = run_chatgpt_with_examples(query, examples, input, num_gen=num_gen, wait_time=wait_time,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature).strip()
            else:
                output = run_chatgpt(query, num_gen=num_gen, wait_time=wait_time, model=model,
                                                   num_tokens_request=num_tokens_request, use_16k=use_16k, temperature=temperature)
            output = output.replace('json', '') # this frequently happens
            facts = json.loads(output.strip())
            run_loop = False
        except json.decoder.JSONDecodeError:
            counter += 1
            time.sleep(1)
            print("Retrying to avoid JsonDecodeError, trial %s ..." % counter)
            print(output)
            if counter == 10:
                print("Exiting after 10 trials")
                sys.exit()
            continue
    return facts


def run_claude(query, max_new_tokens, model_name):
    from anthropic import Anthropic

    if model_name == 'claude-sonnet':
        model_name = "claude-3-sonnet-20240229"
    elif model_name == 'claude-haiku':
        model_name = "claude-3-haiku-20240307"

    client = Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    # print(query)
    message = client.messages.create(
        max_tokens=max_new_tokens,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model_name,
    )
    print(message.content)
    return message.content[0].text


def run_gemini(model, content: str, max_tokens: int = 0):

    try:
        response = model.generate_content(content)
        return response.text
    except Exception as e:
        print(f'{type(e).__name__}: {e}')
        return None


def run_chatgpt(query, num_gen=1, num_tokens_request=1000,
                model='gpt-5-mini', use_16k=False, temperature=1.0, wait_time=1):
    """Harbor-parity port (gpt-5-mini only).

    Uses openai>=1 client which picks up OPENAI_API_KEY + OPENAI_BASE_URL
    from env (parity proxy). Settings mirror the Harbor parity runner
    `adapters/locomo/run_locomo_parity.py` byte-for-byte so both sides issue
    identical API calls:
        reasoning_effort = "minimal"
        max_completion_tokens = max(num_tokens_request, 1024) * 8
        temperature = whatever caller passes (upstream batched mode passes 0)
    """
    from openai import OpenAI, APIError, APIConnectionError, RateLimitError
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    completion = None
    backoff = max(wait_time, 1)
    while completion is None:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                n=num_gen,
                temperature=temperature,
                max_completion_tokens=max(num_tokens_request, 1024) * 8,
                reasoning_effort="minimal",
            )
        except (APIError, APIConnectionError, RateLimitError) as e:
            print(f"OpenAI API error: {e}; waiting {backoff}s")
            time.sleep(backoff)
            backoff *= 2

    return completion.choices[0].message.content
    

def run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1000, use_16k=False, wait_time = 1, temperature=1.0):

    completion = None
    
    messages = [
        {"role": "system", "content": query}
    ]
    for inp, out in examples:
        messages.append(
            {"role": "user", "content": inp}
        )
        messages.append(
            {"role": "system", "content": out}
        )
    messages.append(
        {"role": "user", "content": input}
    )   
    
    while completion is None:
        wait_time = wait_time * 2
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo" if not use_16k else "gpt-3.5-turbo-16k",
                temperature = temperature,
                max_tokens = num_tokens_request,
                n=num_gen,
                messages = messages
            )
        except openai.error.APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except openai.error.ServiceUnavailableError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}; waiting for {wait_time} seconds")
            time.sleep(wait_time)
            pass
    
    return completion.choices[0].message.content
