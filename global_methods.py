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
    """Harbor-parity port.

    Dispatches based on ``model`` prefix:
      - ``codex/<inner_model>``  → shells out to the codex CLI (Harbor parity
        agentic baseline, matches the codex agent run on the Harbor side).
      - anything else            → openai>=1 chat.completions path used by the
        original locomo-parity-agent.
    """
    if model.startswith("codex/"):
        inner_model = model.split("/", 1)[1]
        return _run_codex(query, model=inner_model, wait_time=wait_time)

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


def _run_codex(query, model="gpt-5-mini", wait_time=1, max_retries=3):
    """Shell out to ``codex exec`` and return its final-message text.

    This mirrors what Harbor's ``codex`` agent does on the adapter side, so
    upstream vs Harbor parity for ``model=codex/<inner>`` compares like with
    like. Prompt is piped via stdin to avoid argv length limits.

    Auth: writes an isolated ``CODEX_HOME`` per call with API-key auth, so
    codex never falls back to the user's ChatGPT login (which doesn't have
    gpt-5-mini). Matches harbor/src/harbor/agents/installed/codex.py.
    """
    import json as _json
    import subprocess
    import tempfile

    api_key = os.environ.get("OPENAI_API_KEY") or ""
    base_url = os.environ.get("OPENAI_BASE_URL") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for codex backend")

    # Diagnostic: LOCOMO_CODEX_FILEREAD=1 makes codex read the transcript from a
    # file (like the Harbor agent does) instead of inlining it in the prompt.
    # Used only to isolate the file-read-vs-inline variable in parity analysis.
    context_text = None
    prompt_text = query
    if os.environ.get("LOCOMO_CODEX_FILEREAD"):
        _marker = "Based on the above conversations"
        if _marker in query:
            _ctx, _instr = query.split(_marker, 1)
            context_text = _ctx.strip()
            prompt_text = (
                "The full conversation transcript is in the file `conversation.md` "
                "in your current working directory. Read it carefully, then answer.\n\n"
                "Based on the conversation in conversation.md" + _instr
            )

    backoff = max(wait_time, 1)
    last_error = None
    for attempt in range(1, max_retries + 1):
        with tempfile.TemporaryDirectory(prefix="locomo_codex_") as workdir:
            codex_home = os.path.join(workdir, ".codex")
            os.makedirs(codex_home, exist_ok=True)
            if context_text is not None:
                with open(os.path.join(workdir, "conversation.md"), "w") as f:
                    f.write(context_text)
            with open(os.path.join(codex_home, "auth.json"), "w") as f:
                _json.dump({"OPENAI_API_KEY": api_key}, f)
            if base_url:
                # codex 0.118+ honors openai_base_url only from config.toml, not env.
                with open(os.path.join(codex_home, "config.toml"), "w") as f:
                    f.write(f'openai_base_url = "{base_url}"\n')
            out_path = os.path.join(workdir, "codex_last_message.txt")
            cmd = [
                "codex", "exec",
                "--model", model,
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "--ephemeral",
                "--output-last-message", out_path,
                "-",
            ]
            env = {**os.environ, "CODEX_HOME": codex_home, "OPENAI_API_KEY": api_key}
            if base_url:
                env["OPENAI_BASE_URL"] = base_url
            try:
                proc = subprocess.run(
                    cmd, input=prompt_text, text=True,
                    capture_output=True, cwd=workdir, env=env,
                    timeout=int(os.environ.get("CODEX_TIMEOUT", "900")),
                )
            except subprocess.TimeoutExpired as e:
                last_error = e
                print(f"codex timed out (attempt {attempt}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
            if proc.returncode != 0:
                last_error = RuntimeError(
                    f"codex exec rc={proc.returncode}: {proc.stderr[-500:]}"
                )
                print(f"codex error (attempt {attempt}/{max_retries}): {last_error}")
                time.sleep(backoff)
                backoff *= 2
                continue
            try:
                with open(out_path, encoding="utf-8") as f:
                    return f.read()
            except FileNotFoundError:
                last_error = RuntimeError("codex did not write --output-last-message file")
                print(f"codex missing output (attempt {attempt}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
    raise RuntimeError(f"codex failed after {max_retries} attempts: {last_error}")



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
