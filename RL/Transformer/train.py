# -*- coding: utf-8 -*-
# File: transformer_bo/train.py
from model_dense_enc import build_transformer
from dataset import BilingualDataset, causal_mask
from config import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import warnings
import numpy as np
from tqdm import tqdm
import os
import csv
from pathlib import Path
import math

# HuggingFace datasets / tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

from fvcore.nn import FlopCountAnalysis

from quantize import ftp_modify_model


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, connections_matrix):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask, connections_matrix)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    # full OPUS Books train split, filtered only by seq_len (no subsampling)
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    ds_filtered = [item for item in ds_raw
                   if len(item['translation'][config['lang_src']]) <= config['seq_len']
                   and len(item['translation'][config['lang_tgt']]) <= config['seq_len']]

    tokenizer_src = get_or_build_tokenizer(config, ds_filtered, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_filtered, config['lang_tgt'])

    train_sz = int(0.9 * len(ds_filtered))
    val_sz = len(ds_filtered) - train_sz
    print("length_of_dataset:", len(ds_filtered))

    # deterministic split helps reproducibility
    g = torch.Generator().manual_seed(42)
    train_raw, val_raw = random_split(ds_filtered, [train_sz, val_sz], generator=g)

    train_ds = BilingualDataset(train_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Tokenized max lengths (debug info)
    max_len_src = 0
    max_len_tgt = 0
    for it in ds_filtered:
        src_ids = tokenizer_src.encode(it['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(it['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"])

    init_weights_path = "init_model.pth"
    if os.path.exists(init_weights_path):
        print(f"Loading initial weights from {init_weights_path}")
        # strict load; assumes your seq_len/d_model match config (old way)
        model.load_state_dict(torch.load(init_weights_path))
    else:
        print(f"Saving initial weights at {init_weights_path}")
        torch.save(model.state_dict(), init_weights_path)

    return model


def calculate_perplexity(loss):
    return math.exp(loss)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len,
                   device, print_msg, global_step, writer, connections_matrix, num_examples=2):
    model.eval()
    count = 0
    expected = []
    predicted = []
    total_loss = 0.0
    num_batches = 0

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            num_batches += 1

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device, connections_matrix)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            expected.append(target_text)
            predicted.append(model_out_text)

            # validation loss for perplexity
            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            enc_out = model.encode(encoder_input, encoder_mask, connections_matrix)
            dec_out = model.decode(enc_out, encoder_mask, decoder_input, decoder_mask)
            proj = model.project(dec_out)

            # NOTE: this mirrors your older script (ignore_index from SRC tokenizer)
            loss = nn.CrossEntropyLoss(
                ignore_index=tokenizer_src.token_to_id('[PAD]'),
                label_smoothing=0.1
            )(proj.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_loss += loss.item()

            if count <= num_examples:
                print_msg('-' * console_width)
                print_msg(f"{'SOURCE:':>12}{source_text}")
                print_msg(f"{'TARGET:':>12}{target_text}")
                print_msg(f"{'PREDICTED:':>12}{model_out_text}")
            if count == num_examples:
                print_msg('-' * console_width)

    if writer:
        cer = torchmetrics.CharErrorRate()(predicted, expected)
        wer = torchmetrics.WordErrorRate()(predicted, expected)
        bleu = torchmetrics.BLEUScore()(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.add_scalar('validation wer', wer, global_step)
        writer.add_scalar('validation BLEU', bleu, global_step)
        avg_loss = total_loss / max(1, num_batches)
        writer.add_scalar('validation perplexity', calculate_perplexity(avg_loss), global_step)
        writer.flush()

    return total_loss / max(1, num_batches)


def get_flops(cfg, log:Log=False):
    device = torch.device(DEVICE)
    _, val_loader, tokenizer_src, tokenizer_tgt = get_ds(cfg)

    model = get_model(cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model_filename = latest_weights_file_path(cfg)
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    N = cfg['num_layers']
    default_matrix = np.zeros((N + 1, N + 1), dtype=int)
    for i in range(N):
        default_matrix[i, i+1] = 1
    connections_matrix = torch.tensor(default_matrix, device=device, dtype=torch.int)

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            # === Count FLOPs for this batch ===
            flops_counter = FlopCountAnalysis(
                model, 
                (encoder_input, decoder_input, encoder_mask, decoder_mask, connections_matrix)
            )
            batch_flops = flops_counter.total()

            if log:
                log.println(f"FLOPS: {batch_flops}")
            return batch_flops

###########################################################################################
def train_model(config, validate=False, log:Log=False, precision="ftp32"):
    device = torch.device(DEVICE)
    print("Using device:", device)

    # Default sequential encoder graph if not provided
    N = config['num_layers']
    default_matrix = np.zeros((N + 1, N + 1), dtype=int)
    for i in range(N):
        default_matrix[i, i + 1] = 1
    connections_matrix = config.get("connections_matrix", default_matrix)
    connections_matrix = torch.tensor(connections_matrix, device=device, dtype=torch.int)

    # Weights folder
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    # Data / model
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    if precision != "ftp32":
        raise ("ivalid precision, only ftp32 in training is supported")
    
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Preload OFF for NAS (old way keeps None)
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # NOTE: mirrors your older script (ignore_index from SRC tokenizer)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    # CSV (per run)
    csv_path = os.path.join(model_folder, f"{config['model_folder']}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(['Epoch', 'Validation Perplexity'])

        train_loss = 0
        for epoch in range(initial_epoch, config['num_epochs']):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model.train()
            batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")

            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                enc_out = model.encode(encoder_input, encoder_mask, connections_matrix)
                dec_out = model.decode(enc_out, encoder_mask, decoder_input, decoder_mask)
                proj = model.project(dec_out)

                loss = loss_fn(proj.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                train_loss = loss.item()

                writer.add_scalar('train loss', loss.item(), global_step)
                writer.add_scalar('train perplexity', calculate_perplexity(loss.item()), global_step)
                writer.flush()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            train_perplexity = calculate_perplexity(train_loss)
            writer_csv.writerow([epoch, train_perplexity])
            file.flush()
            

            # save every epoch
            save_path = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, save_path)
            if log:
                log.println(f"epoch [{epoch:2}] saved in [{save_path}], train loss [{train_loss:.4f}]")
                
            if validate and epoch >= validate:
                v_loss = run_validation(
                    model, val_loader, tokenizer_src, tokenizer_tgt, config['seq_len'],
                    device, print, 0, None, connections_matrix
                )
                if log:
                    log.println(f"validate loss: {v_loss}")
                else:
                    print(f"validate loss: {v_loss}")

    return (train_loss, calculate_perplexity(train_loss))


###########################################################################################
def inference_model(cfg, log:Log=False, precision="ftp32"):
    device = torch.device(DEVICE)
    _, val_loader, tokenizer_src, tokenizer_tgt = get_ds(cfg)
    model = get_model(cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    model_filename = latest_weights_file_path(cfg)
    print("Loading:", model_filename)
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    if precision == "ftp64":
        raise ValueError("no bigger than ftp32 supported")
    elif precision == "ftp32":
        pass
    else:
        log.println(f"{model.encoder.layers[0].self_attention_block.w_q.weight}")
        model = ftp_modify_model(model, precision)
        log.println(f"{model.encoder.layers[0].self_attention_block.w_q.weight}")

    # === Connections matrix (sequential encoder by default) ===
    N = cfg['num_layers']
    default_matrix = np.zeros((N + 1, N + 1), dtype=int)
    for i in range(N):
        default_matrix[i, i+1] = 1
    connections_matrix = torch.tensor(default_matrix, device=device, dtype=torch.int)

    # === Run validation once (just inference mode) ===
    avg_loss = run_validation(
        model, val_loader, tokenizer_src, tokenizer_tgt, cfg['seq_len'],
        device, print, 0, None, connections_matrix
    )

    pex = calculate_perplexity(avg_loss)
    if log:
        log.println(f"loss [{avg_loss:.5}], perplexity: [{pex}]")
    return (avg_loss, pex)


if __name__ == '__main__':
    log.println(f"train: ")
    warnings.filterwarnings("ignore")
    cfg = get_config()

    train_model(cfg, validate=False, log=log, precision=PRECISION)
    # get_flops(cfg, log=log)
    log.println("\n\n")
