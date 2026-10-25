"""Check that the official saved-score solver reproduces the allocation and responds to its budget."""
from collections import Counter
from pathlib import Path

import torch
import modelopt.torch.quantization as mtq

from fidelity import write_json


if __name__ == '__main__':
    root = Path('build/autoquant/search_v2')
    state = torch.load(root/'completed_search_state.pt', weights_only=False, map_location='cpu')
    original = mtq.get_auto_quantize_config(state)
    repeated = mtq.get_auto_quantize_config(state, {'effective_bits': 5.5})
    assert original == repeated
    records = []
    for bits in [5.0, 5.5, 6.0]:
        config = mtq.get_auto_quantize_config(state, {'effective_bits': bits})
        counts = Counter()
        for entry in config['quant_cfg']:
            if entry['quantizer_name'].endswith('.weight_quantizer') and entry.get('enable'):
                counts[str(entry['cfg']['num_bits'])] += 1
        records.append(dict(target_bits=bits, weight_quantizer_formats=dict(counts),
                            same_as_original=config == original))
    assert not records[0]['same_as_original'] and records[1]['same_as_original'] and not records[2]['same_as_original']
    write_json(root/'replay_ablation.json', dict(
        original_allocation_reproduced=True, official_api='mtq.get_auto_quantize_config',
        only_changed='effective_bits constraint; identical saved sensitivities',
        new_calibration_or_gpu_inference=False, budgets=records))
