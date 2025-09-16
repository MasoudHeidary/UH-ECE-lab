# -*- coding: utf-8 -*-
# File: transformer_bo/inference.py
from train import *



# from fvcore.nn import FlopCountAnalysis
# def inference_model(config):
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#     # === Load dataset & tokenizers ===
#     _, val_loader, tokenizer_src, tokenizer_tgt = get_ds(cfg)

#     # === Build model ===
#     model = get_model(cfg, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

#     # === Load pretrained checkpoint ===
#     model_filename = latest_weights_file_path(cfg)
#     print("Loading:", model_filename)
#     state = torch.load(model_filename, map_location=device)
#     model.load_state_dict(state["model_state_dict"])

#     # === Connections matrix (sequential encoder by default) ===
#     N = cfg['num_layers']
#     default_matrix = np.zeros((N + 1, N + 1), dtype=int)
#     for i in range(N):
#         default_matrix[i, i+1] = 1
#     connections_matrix = torch.tensor(default_matrix, device=device, dtype=torch.int)

#     # === Run validation once (just inference mode) ===
#     # avg_loss = run_validation(
#     #     model, val_loader, tokenizer_src, tokenizer_tgt, cfg['seq_len'],
#     #     device, print, 0, None, connections_matrix
#     # )
#     avg_loss = run_validation_with_flops(
#         model, val_loader, tokenizer_src, tokenizer_tgt, cfg['seq_len'],
#         device, print, 0, None, connections_matrix
#     )

#     print("Validation Loss:", avg_loss)
#     print("Validation Perplexity:", calculate_perplexity(avg_loss))



# def run_validation_with_flops(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len,
#                               device, print_msg, global_step, writer, connections_matrix, num_examples=2):
#     model.eval()
#     total_flops = 0

#     with torch.no_grad():
#         for batch in validation_ds:
#             encoder_input = batch["encoder_input"].to(device)
#             decoder_input = batch["decoder_input"].to(device)
#             encoder_mask = batch["encoder_mask"].to(device)
#             decoder_mask = batch["decoder_mask"].to(device)

#             # === Count FLOPs for this batch ===
#             flops_counter = FlopCountAnalysis(
#                 model, 
#                 (encoder_input, decoder_input, encoder_mask, decoder_mask, connections_matrix)
#             )
#             batch_flops = flops_counter.total()
#             total_flops += batch_flops

#             # === Run normal forward pass ===
#             encoder_output = model.encode(encoder_input, encoder_mask, connections_matrix)
#             decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
#             output = model.project(decoder_output)

#             # Compute loss / metrics as usual
#             label = batch['label'].to(device)
#             loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'))
#             loss = loss_fn(output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

#             if writer:
#                 writer.add_scalar('validation_loss', loss.item(), global_step)
#             break;

#     avg_flops = total_flops / len(validation_ds)
#     print("Average FLOPs per example:", total_flops)
#     return avg_flops



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    cfg = get_config()
    # inference_model(cfg)
    print(get_flops(cfg))


"""
seq_len: 350 -> FLOPS: 19324748800
seq_len: 250 -> FLOPS: 13342592000

NUM_HEADS: 2 ->
NUM_HEADS: 4 -> FLOPS: 19324748800
"""