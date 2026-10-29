"""
NeuroPredictor web server: same prediction pipeline as gui_app.py, exposed as
a small Flask web app instead of a desktop window, so it can be shared over a
tunnel (e.g. ngrok) while all the heavy computation still runs on this machine.

Usage:
  python web_app.py
Then open http://localhost:5000 in a browser (or tunnel that port with ngrok
to share it: `ngrok http 5000`).
"""
import os
import uuid
import threading

from flask import Flask, jsonify, render_template, request

from predictor_core import parseSequenceInput, runPredictionPipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(SCRIPT_DIR, 'web_jobs')
os.makedirs(JOBS_DIR, exist_ok=True)

app = Flask(__name__)

_jobs = {}
_jobsLock = threading.Lock()


def _setJob(jobId, **fields):
    with _jobsLock:
        _jobs[jobId].update(fields)


def _appendLog(jobId, text):
    with _jobsLock:
        _jobs[jobId]['log'].append(text)


def _runJob(jobId, recordList):
    inputCsvPath = os.path.join(JOBS_DIR, f'{jobId}_input.csv')
    resultCsvPath = os.path.join(JOBS_DIR, f'{jobId}_result.csv')
    try:
        rows = runPredictionPipeline(
            recordList, inputCsvPath, resultCsvPath,
            onLog=lambda text: _appendLog(jobId, text))
        _setJob(jobId, state='done', results=rows)
    except Exception as exc:
        _appendLog(jobId, str(exc))
        _setJob(jobId, state='error', error=str(exc))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def apiPredict():
    payload = request.get_json(silent=True) or {}
    rawText = payload.get('sequences', '')
    recordList = parseSequenceInput(rawText)
    if not recordList:
        return jsonify({'error': 'No valid sequence found in input.'}), 400

    jobId = uuid.uuid4().hex
    with _jobsLock:
        _jobs[jobId] = {'state': 'running', 'log': [], 'results': None, 'error': None}

    thread = threading.Thread(target=_runJob, args=(jobId, recordList), daemon=True)
    thread.start()
    return jsonify({'job_id': jobId})


@app.route('/api/status/<jobId>')
def apiStatus(jobId):
    with _jobsLock:
        job = _jobs.get(jobId)
        if job is None:
            return jsonify({'error': 'Unknown job id'}), 404
        return jsonify(dict(job))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
