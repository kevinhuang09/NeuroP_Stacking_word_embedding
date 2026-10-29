"""
Shared pipeline logic used by both the desktop GUI (gui_app.py) and the web
server (web_app.py): parse sequence input, write the CSV the existing
precompute_embedding.py / predict.py scripts expect, run them in their
respective conda environments, and read back the result CSV.
"""
import os
import csv
import shutil
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRECOMPUTE_SCRIPT = os.path.join(SCRIPT_DIR, 'precompute_embedding.py')
PREDICT_SCRIPT = os.path.join(SCRIPT_DIR, 'predict.py')

CONDA_BIN = shutil.which('conda') or '/home/g503/miniconda3/bin/conda'
BASE_ENV = 'base'
PYCARET_ENV = 'neuropredictor-tool'


def parseSequenceInput(rawText):
    """
    Parse textarea content into [(id, sequence), ...].
    One sequence per line, two supported formats:
      SEQUENCE        -> auto-numbered id (seq1, seq2, ...)
      id,SEQUENCE      -> user-supplied id
    """
    recordList = []
    for i, line in enumerate(rawText.splitlines()):
        line = line.strip()
        if not line:
            continue
        if ',' in line:
            name, seq = line.split(',', 1)
            name, seq = name.strip(), seq.strip()
        else:
            name, seq = f'seq{i + 1}', line
        if seq:
            recordList.append((name, seq))
    return recordList


def writeInputCsv(recordList, csvPath):
    with open(csvPath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'sequence'])
        for name, seq in recordList:
            writer.writerow([name, seq])


def runCondaScript(envName, scriptPath, argList, extraEnv=None):
    """Run a script inside a conda env via `conda run`; returns (returncode, stdout, stderr)."""
    cmd = [CONDA_BIN, 'run', '-n', envName, 'python', scriptPath] + argList
    env = dict(os.environ)
    if extraEnv:
        env.update(extraEnv)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def readResultCsv(csvPath):
    with open(csvPath, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def runPredictionPipeline(recordList, inputCsvPath, resultCsvPath, onLog=None):
    """
    Run the full two-stage pipeline for recordList = [(id, sequence), ...].
    onLog(text) is called with progress messages as they happen (optional).
    Returns the list of result rows (dicts with id/sequence/probability/prediction).
    Raises RuntimeError with the failing script's stderr tail on failure.
    """
    def log(text):
        if onLog:
            onLog(text)

    writeInputCsv(recordList, inputCsvPath)
    log(f'Wrote {len(recordList)} sequence(s) to {inputCsvPath}')

    log('[1/2] Computing LLM embeddings (base env)...')
    code, out, err = runCondaScript(
        BASE_ENV, PRECOMPUTE_SCRIPT, [inputCsvPath], extraEnv={'HF_HUB_OFFLINE': '1'})
    if out.strip():
        log(out.strip())
    if code != 0:
        raise RuntimeError(f'precompute_embedding.py failed (exit={code}):\n{err.strip()[-2000:]}')

    log('[2/2] Running Lv1->Lv2 stacking prediction (neuropredictor-tool env)...')
    code, out, err = runCondaScript(
        PYCARET_ENV, PREDICT_SCRIPT, [inputCsvPath, resultCsvPath])
    if code != 0:
        raise RuntimeError(f'predict.py failed (exit={code}):\n{err.strip()[-2000:]}')

    return readResultCsv(resultCsvPath)
