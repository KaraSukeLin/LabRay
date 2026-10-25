"""Official ModelOpt Weight AutoQuantize, without an externally assigned layer map."""
import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoModelForImageTextToText
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.opt.utils import named_hparams

from analyze import export
from fidelity import write_json

BASE = Path('/root/model/Qwen-Qwen3.8-27B')
# The GDN exclusions follow ModelOpt's official base_disabled_layers recipe.
DISABLED = ['*visual*', '*vision_tower*', '*mtp*', '*lm_head*', '*linear_attn.conv1d*',
            '*linear_attn.in_proj_a*', '*linear_attn.in_proj_b*']
COST_EXCLUDED = ['*visual*', '*vision_tower*', '*mtp*', '*lm_head*']


def validation(model, rows):
    nll, count = 0., 0
    with torch.inference_mode():
        for row in rows:
            ids = torch.tensor([row['input_ids']], device='cuda')
            logits = model(input_ids=ids, use_cache=False).logits
            nll += torch.nn.functional.cross_entropy(logits[:, :-1].float().flatten(0, 1),
                ids[:, 1:].flatten(), reduction='sum').item()
            count += ids.numel() - 1
    return dict(tokens=count, nll=nll, ppl=float(torch.exp(torch.tensor(nll / count))))


def run(args):
    torch.manual_seed(20261016)
    if args.checkpoint.exists():
        raise FileExistsError(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = AutoConfig.from_pretrained(BASE, local_files_only=True)
    if args.probe:
        # This is a synthetic architecture probe; do not carry real source tensors into it.
        config._name_or_path = ''
        text = config.text_config
        for k, v in dict(hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=32, linear_value_head_dim=32).items():
            setattr(text, k, v)
        text.layer_types = ['linear_attention'] * 3 + ['full_attention']
        vision = config.vision_config
        vision.hidden_size, vision.intermediate_size, vision.depth = 64, 128, 1
        vision.num_heads, vision.out_hidden_size = 4, 128
        model = AutoModelForImageTextToText.from_config(config, attn_implementation='eager').to('cuda', torch.bfloat16)
    else:
        model = AutoModelForImageTextToText.from_pretrained(BASE, local_files_only=True,
            dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa')
    model.eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    rows = json.loads(Path('assets/autoquant/calibration.json').read_text())
    heldout = json.loads(Path('assets/autoquant/validation.json').read_text())
    if args.probe:
        rows = [dict(rows[0], input_ids=rows[0]['input_ids'][:16])]
        heldout = rows
    else:
        rows = [dict(r, input_ids=r['input_ids'][:args.sequence_length]) for r in rows]
    batches = [torch.tensor([r['input_ids']], device='cuda') for r in rows]
    before = validation(model, heldout)
    write_json(args.output / 'before.json', before)
    forward_calls = 0
    def forward_step(m, batch):
        nonlocal forward_calls
        forward_calls += 1
        return m(input_ids=batch, use_cache=False).logits
    search_start = time.perf_counter()
    kwargs = dict(constraints={'effective_bits': 5.5, 'cost_model': 'weight',
                              'cost': {'excluded_module_name_patterns': COST_EXCLUDED}},
                  quantization_formats=[mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG],
                  data_loader=batches, forward_step=forward_step,
                  num_calib_steps=len(batches), num_score_steps=1 if args.probe else args.score_steps,
                  disabled_layers=DISABLED, method=args.method, verbose=True,
                  checkpoint=str(args.output / 'search_state.pt'))
    write_json(args.output / 'parameters.json', dict(
        base=str(BASE), base_revision=(BASE / '.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0],
        method=args.method, effective_bits=5.5, cost_model='weight',
        candidates=['NVFP4_DEFAULT_CFG', 'FP8_DEFAULT_CFG', 'BF16 (framework automatic fallback)'],
        disabled_layers=DISABLED, cost_excluded=COST_EXCLUDED,
        calibration_steps=len(batches), score_steps=kwargs['num_score_steps'],
        sequence_length=16 if args.probe else args.sequence_length,
        data_manifest_sha256=hashlib.sha256(Path('assets/autoquant/manifest.json').read_bytes()).hexdigest(),
        packages={p: importlib.metadata.version(p) for p in ['nvidia-modelopt', 'torch', 'transformers']},
        probe=args.probe, gpu=torch.cuda.get_device_name(0)))
    model, state = mtq.auto_quantize(model, **kwargs)
    search_seconds = time.perf_counter() - search_start
    torch.save(state, args.output / 'completed_search_state.pt')
    sensitivities = []
    for group, item in state['candidate_stats'].items():
        for recipe, score_value, raw_score, cost in zip(
                item['formats'], item['scores'], item['raw_scores'], item['costs']):
            sensitivities.append(dict(group=group, modules=';'.join(item['module_names']),
                recipe=str(recipe), score=float(score_value), raw_score=float(raw_score), cost=float(cost),
                selected=str(recipe) == str(state['best']['recipe'][group])))
    export(args.output / 'sensitivities', sensitivities)
    allocation = []
    for name, module in model.named_modules():
        if not hasattr(module, 'quant_recipe'):
            continue
        hp = module.quant_recipe
        if not hasattr(module, 'weight'):
            continue
        allocation.append(dict(module=name, recipe=str(hp), parameters=module.weight.numel(),
            cost_excluded=any(fnmatch.fnmatch(name, p) for p in COST_EXCLUDED)))
    # named_hparams preserves the runtime grouping imposed by ModelOpt for fused kernels.
    groups = [dict(name=n, active=str(h.active), choices=[str(c) for c in h.choices])
              for n, h in named_hparams(model, unique=True) if hasattr(h, 'active')]
    write_json(args.output / 'allocation.json', dict(modules=allocation, groups=groups))
    after = validation(model, heldout)
    write_json(args.output / 'validation.json', dict(before=before, after=after))
    export_hf_checkpoint(model, export_dir=args.checkpoint, dtype=torch.bfloat16)
    weights = sorted(args.checkpoint.glob('*.safetensors'))
    write_json(args.output / 'completed.json', dict(status='ok', probe=args.probe,
        search_seconds=search_seconds, search_gpu_hours=search_seconds / 3600,
        total_seconds=time.perf_counter()-started, forward_calls=forward_calls,
        cost_denominator_parameters=state['cost_denominator'],
        checkpoint=str(args.checkpoint), weight_bytes=sum(p.stat().st_size for p in weights),
        best=json.loads(json.dumps(state.get('best', {}), default=str))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--method', choices=['kl_div', 'aumann_shapley', 'gradient'], default='kl_div')
    p.add_argument('--sequence-length', type=int, default=512)
    p.add_argument('--score-steps', type=int, default=8)
    run(p.parse_args())
