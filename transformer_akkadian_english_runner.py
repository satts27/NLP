import math
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sacrebleu import corpus_bleu, corpus_chrf
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>", "<sep>"]
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.csv"
BEST_PATH = ROOT / "transformer_akkadian_best.pt"
HISTORY_PATH = ROOT / "transformer_history.csv"
ATTENTION_MAP_PATH = ROOT / "transformer_attention_map.png"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace('"', "")
    text = text.replace("<gap>", " <sep> ")
    return text


def train_bpe(texts, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def encode_bpe(texts, tokenizer: Tokenizer):
    return [enc.ids for enc in tokenizer.encode_batch(texts)]


def filter_by_length(df: pd.DataFrame, max_len: int = 80) -> pd.DataFrame:
    df = df.copy()
    df["src_len"] = df["src_ids"].apply(len)
    df["tgt_len"] = df["tgt_ids"].apply(len)
    df = df[(df["src_len"] <= max_len) & (df["tgt_len"] <= max_len)]
    df = df.sort_values(by="src_len").reset_index(drop=True)
    return df


class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.src = df["src_ids"].tolist()
        self.tgt = df["tgt_ids"].tolist()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.tensor(self.src[idx]), torch.tensor(self.tgt[idx])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerMT(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 512,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(
            src_vocab_size, d_model, padding_idx=src_pad_idx
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=tgt_pad_idx
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        return src == self.src_pad_idx

    def make_tgt_key_padding_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        return tgt == self.tgt_pad_idx

    @staticmethod
    def make_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        src_pad_mask = self.make_src_key_padding_mask(src)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.transformer_encoder(src_emb, src_key_padding_mask=src_pad_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor):
        tgt_len = tgt.size(1)
        tgt_causal_mask = self.make_causal_mask(tgt_len, tgt.device)
        tgt_pad_mask = self.make_tgt_key_padding_mask(tgt)
        src_pad_mask = self.make_src_key_padding_mask(src)

        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.fc_out(out)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        tgt_input = tgt[:, :-1]
        memory = self.encode(src)
        return self.decode(tgt_input, memory, src)


def translate_greedy(
    model, sentence, src_tokenizer, tgt_tokenizer, device, sos_idx, eos_idx, max_len=50
):
    model.eval()
    src_ids = src_tokenizer.encode(sentence).ids
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        memory = model.encode(src_tensor)

    generated = [sos_idx]

    for _ in range(max_len):
        tgt_tensor = torch.tensor(generated).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.decode(tgt_tensor, memory, src_tensor)
        next_id = logits[0, -1].argmax().item()
        if next_id == eos_idx:
            break
        generated.append(next_id)

    return tgt_tokenizer.decode(generated[1:])


def translate_beam(
    model,
    sentence,
    src_tokenizer,
    tgt_tokenizer,
    device,
    sos_idx,
    eos_idx,
    beam_size=4,
    max_len=50,
    length_penalty=0.7,
):
    model.eval()
    src_ids = src_tokenizer.encode(sentence).ids
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        memory = model.encode(src_tensor)

    beams = [(0.0, [sos_idx])]
    completed = []

    for _ in range(max_len):
        all_candidates = []

        for log_prob, ids in beams:
            tgt_tensor = torch.tensor(ids).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model.decode(tgt_tensor, memory, src_tensor)
            log_probs = F.log_softmax(logits[0, -1], dim=-1)
            topk_lp, topk_idx = log_probs.topk(beam_size)

            for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                new_log_prob = log_prob + lp
                new_ids = ids + [idx]
                if idx == eos_idx:
                    pen_score = new_log_prob / (len(ids) ** length_penalty)
                    completed.append((pen_score, ids[1:]))
                else:
                    all_candidates.append((new_log_prob, new_ids))

        beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        if not beams:
            break

    if completed:
        best_ids = max(completed, key=lambda x: x[0])[1]
    else:
        best_ids = beams[0][1][1:]

    return tgt_tokenizer.decode(best_ids)


def evaluate_model(
    model,
    df_split,
    src_tokenizer,
    tgt_tokenizer,
    device,
    sos_idx,
    eos_idx,
    beam=True,
    beam_size=4,
):
    preds, refs = [], []

    if beam:
        decode_fn = lambda s: translate_beam(
            model,
            s,
            src_tokenizer,
            tgt_tokenizer,
            device,
            sos_idx,
            eos_idx,
            beam_size=beam_size,
        )
    else:
        decode_fn = lambda s: translate_greedy(
            model, s, src_tokenizer, tgt_tokenizer, device, sos_idx, eos_idx
        )

    for _, row in df_split.iterrows():
        src = row["source"]
        ref = row["translation"].replace("<sep>", " ").strip()
        pred = decode_fn(src).replace("<sep>", " ").strip()
        preds.append(pred)
        refs.append(ref)

    bleu = corpus_bleu(preds, [refs]).score
    chrf = corpus_chrf(preds, [refs]).score
    return bleu, chrf


def get_cross_attention(
    model,
    sentence,
    src_tokenizer,
    tgt_tokenizer,
    device,
    sos_idx,
    eos_idx,
    max_len=50,
):
    model.eval()

    src_ids = src_tokenizer.encode(sentence).ids
    src_tokens = [src_tokenizer.id_to_token(i) for i in src_ids]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    cross_attn_weights = []

    def hook_fn(module, input, output):
        if output[1] is not None:
            cross_attn_weights.append(output[1].detach().cpu())

    last_decoder_layer = model.transformer_decoder.layers[-1]
    hook = last_decoder_layer.multihead_attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        memory = model.encode(src_tensor)

    generated = [sos_idx]
    attn_rows = []

    for _ in range(max_len):
        cross_attn_weights.clear()
        tgt_t = torch.tensor(generated).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.decode(tgt_t, memory, src_tensor)
        if not cross_attn_weights:
            break
        attn_last = cross_attn_weights[0][0, -1, : len(src_ids)].numpy()
        next_id = logits[0, -1].argmax().item()
        if next_id == eos_idx:
            break
        generated.append(next_id)
        attn_rows.append(attn_last)

    hook.remove()

    tgt_tokens = [tgt_tokenizer.id_to_token(i) for i in generated[1:]]
    attn_matrix = np.array(attn_rows)
    return tgt_tokens, src_tokens, attn_matrix


def plot_attention(sentence, tgt_tokens, src_tokens, attn_matrix):
    if attn_matrix.size == 0:
        print("Skipping attention plot because no attention weights were captured.", flush=True)
        return

    fig, ax = plt.subplots(
        figsize=(max(8, len(src_tokens)), max(6, max(1, len(tgt_tokens)) // 2))
    )
    im = ax.matshow(attn_matrix, cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90, fontsize=9)
    ax.set_yticklabels(tgt_tokens, fontsize=9)
    ax.set_xlabel("Source (Akkadian BPE tokens)")
    ax.set_ylabel("Target (English BPE tokens)")
    ax.set_title(f'Cross-Attention (last decoder layer): "{sentence[:60]}"')
    plt.tight_layout()
    plt.savefig(ATTENTION_MAP_PATH, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[["transliteration", "translation"]].dropna()
    df = df.rename(columns={"transliteration": "source"})
    df["source"] = df["source"].apply(clean_text)
    df["translation"] = df["translation"].apply(clean_text)
    print(f"Total samples: {len(df)}", flush=True)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}",
        flush=True,
    )

    print("Training source (Akkadian) BPE tokenizer...", flush=True)
    src_tokenizer = train_bpe(train_df["source"].tolist(), vocab_size=4000)
    print("Training target (English) BPE tokenizer...", flush=True)
    tgt_tokenizer = train_bpe(train_df["translation"].tolist(), vocab_size=6000)

    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    print(
        f"Source BPE vocab: {src_vocab_size} | Target BPE vocab: {tgt_vocab_size}",
        flush=True,
    )

    src_pad_idx = src_tokenizer.token_to_id("<pad>")
    tgt_pad_idx = tgt_tokenizer.token_to_id("<pad>")
    sos_idx = tgt_tokenizer.token_to_id("<sos>")
    eos_idx = tgt_tokenizer.token_to_id("<eos>")

    for split_df in [train_df, val_df, test_df]:
        split_df["src_ids"] = encode_bpe(split_df["source"].tolist(), src_tokenizer)
        tgt_ids_raw = encode_bpe(split_df["translation"].tolist(), tgt_tokenizer)
        split_df["tgt_ids"] = [[sos_idx] + ids + [eos_idx] for ids in tgt_ids_raw]

    train_df = filter_by_length(train_df, 80)
    val_df = filter_by_length(val_df, 80)
    test_df = filter_by_length(test_df, 80)
    print(
        f"After filtering -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}",
        flush=True,
    )

    def collate_fn(batch):
        src_seqs, tgt_seqs = zip(*batch)
        src_padded = pad_sequence(
            src_seqs, batch_first=True, padding_value=src_pad_idx
        )
        tgt_padded = pad_sequence(
            tgt_seqs, batch_first=True, padding_value=tgt_pad_idx
        )
        return src_padded, tgt_padded

    batch_size = 32
    train_loader = DataLoader(
        TranslationDataset(train_df),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TranslationDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(
        f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {math.ceil(len(test_df)/batch_size)}",
        flush=True,
    )

    d_model = 256
    nhead = 8
    num_enc_layers = 3
    num_dec_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    label_smoothing = 0.1
    warmup_steps = 4000
    n_epochs = 300
    clip = 1.0
    patience = 10

    model = TransformerMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_enc_layers,
        num_decoder_layers=num_dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=512,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    def noam_lambda(step: int) -> float:
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_pad_idx,
        label_smoothing=label_smoothing,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}", flush=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    global_step = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]"):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt)
            tgt_labels = tgt[:, 1:].reshape(-1)
            logits_flat = logits.reshape(-1, tgt_vocab_size)

            loss = criterion(logits_flat, tgt_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            global_step += 1
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                logits = model(src, tgt)
                tgt_flat = tgt[:, 1:].reshape(-1)
                log_flat = logits.reshape(-1, tgt_vocab_size)
                val_loss += criterion(log_flat, tgt_flat).item()

        avg_val_loss = val_loss / len(val_loader)
        val_bleu, val_chrf = evaluate_model(
            model,
            val_df,
            src_tokenizer,
            tgt_tokenizer,
            device,
            sos_idx,
            eos_idx,
            beam=False,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_bleu": val_bleu,
                "val_chrf": val_chrf,
                "lr": current_lr,
                "step": global_step,
            }
        )

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val BLEU: {val_bleu:.2f} | "
            f"Val chrF++: {val_chrf:.2f} | LR: {current_lr:.6f} | Step: {global_step}",
            flush=True,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_PATH)
            print(f"New best model saved (val_loss={best_val_loss:.4f})", flush=True)
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})", flush=True)
            if patience_counter >= patience:
                print("Early stopping triggered", flush=True)
                break

    print("Training complete.", flush=True)

    model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    test_bleu_greedy, test_chrf_greedy = evaluate_model(
        model,
        test_df,
        src_tokenizer,
        tgt_tokenizer,
        device,
        sos_idx,
        eos_idx,
        beam=False,
    )
    test_bleu_beam, test_chrf_beam = evaluate_model(
        model,
        test_df,
        src_tokenizer,
        tgt_tokenizer,
        device,
        sos_idx,
        eos_idx,
        beam=True,
        beam_size=4,
    )

    print("=" * 52, flush=True)
    print("  TEST SET RESULTS  (Vanilla Transformer + BPE)", flush=True)
    print("=" * 52, flush=True)
    print(
        f"  Greedy  - BLEU: {test_bleu_greedy:.2f}  chrF++: {test_chrf_greedy:.2f}",
        flush=True,
    )
    print(
        f"  Beam-4  - BLEU: {test_bleu_beam:.2f}  chrF++: {test_chrf_beam:.2f}",
        flush=True,
    )
    print("=" * 52, flush=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)
    print(history_df.to_string(index=False), flush=True)

    sample_sentence = test_df.iloc[0]["source"]
    tgt_tok, src_tok, attn = get_cross_attention(
        model,
        sample_sentence,
        src_tokenizer,
        tgt_tokenizer,
        device,
        sos_idx,
        eos_idx,
    )
    plot_attention(sample_sentence, tgt_tok, src_tok, attn)

    print("=" * 70, flush=True)
    print("  SAMPLE TRANSLATIONS (Beam-4)", flush=True)
    print("=" * 70, flush=True)
    for i in range(min(5, len(test_df))):
        row = test_df.iloc[i]
        src = row["source"]
        ref = row["translation"].replace("<sep>", " ").strip()
        pred = translate_beam(
            model,
            src,
            src_tokenizer,
            tgt_tokenizer,
            device,
            sos_idx,
            eos_idx,
            beam_size=4,
        ).replace("<sep>", " ").strip()
        print(f"\n[{i + 1}]", flush=True)
        print(f"  SRC  : {src}", flush=True)
        print(f"  REF  : {ref}", flush=True)
        print(f"  PRED : {pred}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
