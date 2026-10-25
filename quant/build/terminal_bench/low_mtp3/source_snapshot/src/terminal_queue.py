"""Durable trial identities and atomic claims; terminal outcomes are immutable."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time

TERMINAL={'PASS','MODEL_FAIL','TIMEOUT','API_FAILURE','ENVIRONMENT_FAIL','VERIFIER_FAIL','INTERRUPTED'}


class Queue:
    def __init__(self,path,protocol):
        self.db=sqlite3.connect(path,timeout=30,isolation_level=None)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS trials (
            id TEXT PRIMARY KEY, model TEXT NOT NULL, task TEXT NOT NULL, mode INTEGER NOT NULL,
            seed INTEGER NOT NULL, attempt INTEGER NOT NULL, protocol TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING', gpu INTEGER, owner_pid INTEGER,
            owner_start TEXT, started REAL, finished REAL, result TEXT,
            UNIQUE(model,task,mode,seed,attempt,protocol))''')
        self.protocol=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('protocol',self.protocol))
        if self.db.execute('SELECT value FROM metadata WHERE key=?',('protocol',)).fetchone()[0]!=self.protocol:
            raise ValueError('Resume protocol identity changed')

    def enqueue(self,models,tasks,seed,modes=(False,True)):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for model in models:
                for task in tasks:
                    for mode in modes:
                        identity=[model,task,int(mode),seed,0,self.protocol]
                        key=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
                        self.db.execute('''INSERT OR IGNORE INTO trials
                            (id,model,task,mode,seed,attempt,protocol) VALUES (?,?,?,?,?,?,?)''',[key,*identity])
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def claim(self,gpu,owner_pid,owner_start,ready_models,preferred=None):
        if not ready_models: return None
        self.db.execute('BEGIN IMMEDIATE')
        try:
            # One active trial on each GPU. Loading a service does not create extra work.
            if self.db.execute("SELECT 1 FROM trials WHERE gpu=? AND status='RUNNING'",(gpu,)).fetchone():
                raise RuntimeError(f'GPU {gpu} already has a running trial')
            placeholders=','.join('?' for _ in ready_models)
            rows=self.db.execute(f"SELECT * FROM trials WHERE status='PENDING' AND model IN ({placeholders}) ORDER BY id",ready_models).fetchall()
            if not rows:
                self.db.execute('COMMIT')
                return None
            candidates=[row for row in rows if row['model']==preferred]
            if not candidates:
                remaining={model:sum(row['model']==model for row in rows) for model in ready_models}
                active={model:self.db.execute("SELECT count(*) FROM trials WHERE model=? AND status='RUNNING'",(model,)).fetchone()[0] for model in ready_models}
                model=max(ready_models,key=lambda model:(remaining[model]/(active[model]+1),model))
                candidates=[row for row in rows if row['model']==model]
            row=dict(candidates[0])
            self.db.execute("UPDATE trials SET status='RUNNING',gpu=?,owner_pid=?,owner_start=?,started=? WHERE id=?",
                (gpu,owner_pid,owner_start,time.time(),row['id']))
            self.db.execute('COMMIT')
            return dict(self.db.execute('SELECT * FROM trials WHERE id=?',(row['id'],)).fetchone())
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def finish(self,key,status,result):
        if status not in TERMINAL: raise ValueError(status)
        changed=self.db.execute("UPDATE trials SET status=?,result=?,finished=? WHERE id=? AND status='RUNNING'",
            (status,str(result),time.time(),key)).rowcount
        if changed!=1: raise RuntimeError('Only a running trial can be finished; results cannot be overwritten')

    def snapshot(self):
        return [dict(row) for row in self.db.execute('SELECT * FROM trials ORDER BY id')]

    def interrupted_owners(self):
        missing=[]
        for row in self.db.execute("SELECT * FROM trials WHERE status='RUNNING'"):
            try:
                start=Path(f'/proc/{row["owner_pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]
            except (FileNotFoundError,ProcessLookupError): start=None
            if start!=row['owner_start']: missing.append(dict(row))
        # Report only: absence does not authorize a retry or loss of partial evidence.
        return missing
