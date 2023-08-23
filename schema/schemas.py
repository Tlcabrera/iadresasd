def one_prompt(prompt) -> dict:
    return {
        "text": prompt["text"]
    }

def list_all(prompts) -> list:
    return [one_prompt(prompt) for prompt in prompts]


def one_embedding(embed)->dict:
    return{
        "embed":embed["text"]
    }