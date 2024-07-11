from huggingface_hub import login

def huggingface_login():
    access_token_read = "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"
    login(token=access_token_read)