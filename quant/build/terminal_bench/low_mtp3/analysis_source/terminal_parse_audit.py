"""Replay the original Terminus JSON parser; preserve model formatting failures."""

from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
from terminal_bench import read,save
from terminal_low import OUTPUT


def main():
    parser=TerminusJSONPlainParser()
    rows=[]
    for path in sorted((OUTPUT/'services').rglob('audit/*.json')):
        event=read(path)
        response=event.get('response')
        if not isinstance(response,dict) or not response.get('choices'):continue
        key=event['identity']['x-terminal-identity'].rsplit('/',1)[-1]
        content=response['choices'][0]['message'].get('content') or ''
        parsed=parser.parse_response(content)
        rows.append(dict(trial=key,model=event['model'],source=str(path),started_at=event['started_at'],
            parse_error=parsed.error,parse_warning=parsed.warning,commands=len(parsed.commands),task_complete=parsed.is_task_complete,
            api_seconds=event['elapsed_seconds']))
    rows.sort(key=lambda row:row['started_at'])
    counts={model:dict(replies=sum(row['model']==model for row in rows),
        parse_errors=sum(bool(row['parse_error']) for row in rows if row['model']==model),
        parse_warnings=sum(bool(row['parse_warning']) for row in rows if row['model']==model)) for model in {row['model'] for row in rows}}
    save(OUTPUT/'TERMINUS_PARSE_AUDIT.json',dict(rows=rows,by_model=counts,
        note='Unmodified official parser replay. Formatting errors are model diagnostics; original native trial rewards and outcomes remain authoritative.'))
    print(counts)


if __name__=='__main__':main()
