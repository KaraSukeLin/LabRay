import sys
from pathlib import Path
sys.path.insert(0,str(Path('experiments/orsharc/src').resolve()))
from b_orsharc.cli import main
from b_orsharc.common import write_json
root=Path('experiments/orsharc/artifacts/smoke/phase3-20261023').resolve()
original=Path.open
accesses=[]
def guarded(path,*args,**kwargs):
 if 'sealed' in path.parts: raise AssertionError('inference opened sealed gold')
 accesses.append(str(path))
 return original(path,*args,**kwargs)
Path.open=guarded
argv=['infer','--model','elephant','--checkpoint',str(root/'production-training/native/merged'),'--partition','train_256','--limit','8','--out',str(root/'cli-inference')]
try:
 main(argv)
 main(argv)
finally: Path.open=original
write_json(root/'isolation.json',{'status':'PASS','sealed_opens':0,'files_opened':sorted(set(accesses)),'command':argv,'repeat':'same identity skipped'})
main(['report','--run',str(root/'cli-inference'),'--out',str(root/'cli-report')])
