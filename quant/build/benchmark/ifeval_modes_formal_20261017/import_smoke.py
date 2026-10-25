"""Verify and import the completed first-stage IFEval smoke into this full run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark import BASE, score, scorers
from ifeval_modes import extract_answer, read_records, render_prompt, sampling_values, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', type=Path, default=ROOT/'build/benchmark/ifeval_modes_smoke_v1')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    smoke_exp = json.loads((args.smoke/'experiment.json').read_text())
    target_exp = json.loads((args.output/'experiment.json').read_text())
    for name in ['config', 'config_sha256', 'models', 'modes', 'seeds', 'dataset_sha256',
                 'model_versions', 'environment']:
        assert smoke_exp[name] == target_exp[name], f'Cannot reuse smoke: {name} differs'
    assert set(smoke_exp['question_ids']) <= set(target_exp['question_ids'])
    assert set(smoke_exp['files_sha256']) - {'/root/projects/quant/src/ifeval_modes.py'} == \
           set(target_exp['files_sha256']) - {'/root/projects/quant/src/ifeval_modes.py'}
    for name in smoke_exp['files_sha256']:
        if name != '/root/projects/quant/src/ifeval_modes.py':
            assert smoke_exp['files_sha256'][name] == target_exp['files_sha256'][name]
    validation = ROOT/'build/benchmark/ifeval_modes_validation'
    simplification = json.loads((validation/'simplification_audit.json').read_text())
    replay = json.loads((validation/'official_replay/audit.json').read_text())
    assert simplification['passed'] and simplification['inference_prompt_sampling_parser_grader_ast_unchanged']
    assert replay['passed'] and replay['official_cli_responses'] == 60
    assert sha(validation/'ifeval_modes_smoke_source.py') == smoke_exp['files_sha256']['/root/projects/quant/src/ifeval_modes.py']
    expected_rows = len(smoke_exp['models']) * len(smoke_exp['modes']) * len(smoke_exp['seeds']) * len(smoke_exp['questions'])
    processor_class = __import__('transformers', fromlist=['AutoProcessor']).AutoProcessor
    processor = processor_class.from_pretrained(BASE, local_files_only=True)
    ifeval, mmmu = scorers()
    target_questions = {r['id']: r for r in target_exp['questions']}
    prompt_ids = {}
    total = 0
    args.output.mkdir(parents=True, exist_ok=True)
    for model in smoke_exp['models']:
        dest = args.output/model
        assert not (dest/'questions.jsonl').exists(), f'Formal rows already exist for {model}'
        source_rows = read_records(args.smoke/model/'questions.jsonl')
        assert len(source_rows) == len(smoke_exp['question_ids']) * len(smoke_exp['modes']) * len(smoke_exp['seeds'])
        (dest/'attempts').mkdir(parents=True, exist_ok=True)
        imported_attempts = {}
        rows = []
        for row in source_rows:
            settings = target_exp['config']['modes'][row['mode']]
            question = target_questions[row['id']]
            assert question == next(q for q in smoke_exp['questions'] if q['id'] == row['id'])
            assert row['sampling'] == row['engine_sampling'] == sampling_values(settings, row['seed'])
            assert row['chat_template_kwargs'] == dict(enable_thinking=settings['enable_thinking'],
                preserve_thinking=True, **({'reasoning_effort': settings['reasoning_effort']}
                    if settings['enable_thinking'] else {}))
            prompt = render_prompt(processor, question, settings)
            ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
            assert row['prompt'] == prompt and row['prompt_token_ids'] == ids
            key = row['mode'], row['id']
            assert prompt_ids.setdefault(key, ids) == ids
            assert processor.tokenizer.decode(row['token_ids'], skip_special_tokens=True) == row['raw_output']
            parsed = extract_answer(processor.tokenizer,
                SimpleNamespace(text=row['raw_output'], token_ids=row['token_ids']),
                row['prompt_token_ids'], settings)
            assert all(row[k] == v for k, v in parsed.items())
            graded = score(question, parsed['final_answer'], ifeval, mmmu)
            assert all(row[k] == v for k, v in graded.items())
            attempt_name = row['attempt']
            if attempt_name not in imported_attempts:
                source_attempt = args.smoke/model/'attempts'/attempt_name
                attempt = json.loads(source_attempt.read_text())
                dest_name = 'reused_'+attempt_name
                attempt['reused_from'] = str(source_attempt.relative_to(ROOT))
                attempt['reuse_verified_utc'] = datetime.now(timezone.utc).isoformat()
                (dest/'attempts'/dest_name).write_text(json.dumps(attempt, indent=2)+'\n')
                imported_attempts[attempt_name] = dest_name
            row['attempt'] = imported_attempts[attempt_name]
            row['final_answer_tokens'] = len(processor.tokenizer.encode(parsed['final_answer'], add_special_tokens=False))
            row['reused_from_smoke'] = True
            rows.append(row)
            total += 1
        temporary = dest/'questions.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        temporary.replace(dest/'questions.jsonl')
    assert total == expected_rows
    digest = hashlib.sha256((args.output/'experiment.json').read_bytes()).hexdigest()
    audit = dict(passed=True, inference_started=False, imported_rows=total,
        models=len(smoke_exp['models']), modes=smoke_exp['modes'], seeds=smoke_exp['seeds'],
        questions=smoke_exp['question_ids'], official_google_cli_replays=replay['official_cli_responses'],
        parser_replays=total, actual_engine_sampling_matches=total,
        official_score_matches=total, prompts_and_token_ids_match=total,
        smoke_source_hash_verified=True, model_data_config_environment_identity_verified=True,
        target_experiment_sha256=digest)
    (args.output/'smoke_reuse_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
