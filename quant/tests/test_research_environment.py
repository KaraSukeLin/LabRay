import subprocess
import sys


def test_spawned_interpreter_can_import_the_shared_cutlass_package():
    result = subprocess.run([sys.executable, '-c', 'import cutlass; print(cutlass.__file__)'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'nvidia_cutlass_dsl' in result.stdout
