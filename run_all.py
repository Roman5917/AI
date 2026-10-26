"""Run all 20 experiments sequentially in isolated processes; resume verified runs."""
import common as c
import argparse
import importlib.metadata
import os
import platform
import subprocess
import sys
import time


def environment():
    cpu = platform.processor()
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'openpyxl',
                'torch', 'tensorflow', 'paddlepaddle', 'mindspore']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(cpu=cpu, logical_cpus=os.cpu_count(), platform=platform.platform(),
                python=sys.version, executable=sys.executable, packages=versions,
                protocol=c.PROTOCOL, seeds=list(c.SEEDS),
                process_isolation='One process per framework/seed; sequential, no simultaneous training',
                timing_limitations='Uncontrolled OS background load, thermal/power state and framework-specific execution/Adam kernels; no universal speed ranking')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', choices=list(c.FRAMEWORKS), action='append', help='Optional subset for diagnosis')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('This protocol targets Python 3.12')
    c.OUTPUT_DIR.mkdir(exist_ok=True)
    logs = c.OUTPUT_DIR / 'logs'; logs.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(c.ROOT / 'prepare_data.py')], cwd=c.ROOT, check=True)
    data = c.load_data()
    c.write_json(c.OUTPUT_DIR / 'environment.json', environment())
    failures = []
    # Interleave frameworks by seed, rather than running all five of one first.
    for seed in c.SEEDS:
        for name in (args.framework or c.FRAMEWORKS):
            try:
                if c.run_path(name, seed).exists():
                    c.validate_run(name, seed, data)
                    print(f'SKIP verified: {name} seed={seed}', flush=True)
                    continue
                stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
                log_path = logs / f'{name}_seed_{seed}_{stamp}.log'
                print(f'RUN {name} seed={seed}; log={log_path.name}', flush=True)
                env = os.environ.copy(); env['PYTHONIOENCODING'] = 'utf-8'
                with log_path.open('w', encoding='utf-8') as log:
                    completed = subprocess.run([sys.executable, '-u', str(c.ROOT / c.FRAMEWORKS[name]),
                                                '--seed', str(seed)], cwd=c.ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f'exit={completed.returncode}; see {log_path}')
                c.validate_run(name, seed, data)
                print(f'OK {name} seed={seed}', flush=True)
            except Exception as exc:
                failures.append(dict(framework=name, seed=seed, error=str(exc)))
                print(f'FAILED {name} seed={seed}: {exc}', flush=True)
    c.write_json(c.OUTPUT_DIR / 'orchestration_status.json', dict(failures=failures,
                 requested_frameworks=list(args.framework or c.FRAMEWORKS), requested_seeds=list(c.SEEDS)))
    if failures:
        raise SystemExit(f'{len(failures)} run(s) failed; no complete comparison will be claimed.')
    if args.framework:
        print('Selected framework runs finished; full comparison requires all four.')
        return
    for script in ('compare_results.py', 'verify_results.py'):
        subprocess.run([sys.executable, str(c.ROOT / script)], cwd=c.ROOT, check=True)


if __name__ == '__main__':
    main()
