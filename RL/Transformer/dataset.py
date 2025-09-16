import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Store raw IDs (ints)
        self.sos_id = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_id = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_id = tokenizer_tgt.token_to_id("[PAD]")

        self.sos_token = torch.tensor([self.sos_id], dtype=torch.int64)
        self.eos_token = torch.tensor([self.eos_id], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pair = self.ds[idx]
        src_text = pair["translation"][self.src_lang]
        tgt_text = pair["translation"][self.tgt_lang]

        # Tokenize
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Padding sizes
        enc_pad = self.seq_len - len(enc_input_tokens) - 2  # <s> ... </s>
        dec_pad = self.seq_len - len(dec_input_tokens) - 1  # <s> ... (no </s> on decoder input)

        if enc_pad < 0 or dec_pad < 0:
            raise ValueError("Sentence is too long for configured seq_len")

        # Encoder input: <s> tokens </s> [PAD...]
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full((enc_pad,), self.pad_id, dtype=torch.int64),
            ],
            dim=0,
        )

        # Decoder input: <s> tokens [PAD...]
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.full((dec_pad,), self.pad_id, dtype=torch.int64),
            ],
            dim=0,
        )

        # Label: tokens </s> [PAD...]
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full((dec_pad,), self.pad_id, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Boolean masks
        enc_nonpad = (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)
        dec_nonpad = (decoder_input != self.pad_id).unsqueeze(0)              # (1,seq_len)
        dec_causal = causal_mask(decoder_input.size(0))                       # (1,seq_len,seq_len)
        decoder_mask = dec_nonpad & dec_causal                                # (1,seq_len,seq_len)

        return {
            "encoder_input": encoder_input,      # (seq_len)
            "decoder_input": decoder_input,      # (seq_len)
            "encoder_mask": enc_nonpad,          # (1,1,seq_len) boolean
            "decoder_mask": decoder_mask,        # (1,seq_len,seq_len) boolean
            "label": label,                      # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size: int) -> torch.Tensor:
    # True where allowed (lower triangle incl. diag), False where masked
    upper = torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)
    return ~upper
