"""Launch the study with isolated cache, thread and native-library settings."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

env=os.environ.copy()
env.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'tb-weather-mplconfig'))
env.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'tb-weather-xdg-cache'))
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    env[name]='1'
if sys.platform=='darwin':
    spec=importlib.util.find_spec('sklearn')
    if spec and spec.origin:
        native=Path(spec.origin).parent/'.dylibs'
        if (native/'libomp.dylib').exists():
            env['DYLD_LIBRARY_PATH']=str(native)+(os.pathsep+env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '')
env['PYTHONPATH']=str(Path(__file__).resolve().parent)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
raise SystemExit(subprocess.call([sys.executable,'-m','study.checks',*sys.argv[1:]],env=env))
