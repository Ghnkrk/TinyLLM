import torch
import sentencepiece as spm
from Architecture import DecoderOnlyTransformer, params
import argparse
import torch.nn.functional as F

vocab_size = params["vocab_size"]
d_model = params["d_model"]
num_layers = params["num_layers"]
num_heads = params["num_heads"]
d_ffn = params["d_ffn"]
max_len = params["max_len"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DecoderOnlyTransformer(vocab_size, d_model, num_layers, num_heads, d_ffn, max_len)
checkpoint = torch.load("final_model.pt", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

sp = spm.SentencePieceProcessor()
sp.load("tinystories_sp.model")

def top_k_filter(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_value = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_value, torch.full_like(logits, -1e10), logits)

def top_p_filter(logits, p):
    if p == 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    mask = cumulative_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(mask, -1e10)
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)

@torch.no_grad()
def generate(
    model, 
    sp,
    prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0
):
    model.eval()

    # Encode prompt to token IDs
    input_ids = torch.tensor(sp.encode(prompt, out_type=int), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):

        # Forward pass
        logits = model(input_ids)  # (1, T, vocab)
        logits = logits[:, -1, :]  # last token

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                logits[0, token_id] /= repetition_penalty

        # Apply temperature
        logits = logits / max(temperature, 1e-6)

        # Apply top-k
        logits = top_k_filter(logits, top_k)

        # Apply top-p
        logits = top_p_filter(logits, top_p)

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode final output
    return sp.decode(input_ids[0].tolist())






if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate text using a trained model.')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for text generation.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation.')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k filtering.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p filtering.')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty.')

    args = parser.parse_args()

    # Call the generate function with parsed arguments
    output = generate(
        model=model,
        sp=sp,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )
    print(device)
    print(output)