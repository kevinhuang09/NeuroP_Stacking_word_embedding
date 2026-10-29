"""
NeuroPredictor desktop GUI: enter amino acid sequences, click Predict to see a
"Predicting..." status while precompute_embedding.py (base env) -> predict.py
(neuropredictor-tool env) run, then show each sequence's probability and
positive(neuropeptide)/negative verdict.

Usage:
  python gui_app.py
(Run with any Python that has tkinter; the GUI itself needs neither torch nor
 pycaret, it just shells out to `conda run` for those two existing scripts.)
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from predictor_core import parseSequenceInput, runPredictionPipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GUI_INPUT_CSV = os.path.join(SCRIPT_DIR, 'gui_input.csv')
GUI_RESULT_CSV = os.path.join(SCRIPT_DIR, 'gui_result.csv')


class NeuroPredictorApp:
    def __init__(self, root):
        self.root = root
        root.title('NeuroPredictor - Neuropeptide Prediction')
        root.geometry('760x560')

        self.resultQueue = queue.Queue()

        mainFrame = ttk.Frame(root, padding=12)
        mainFrame.pack(fill='both', expand=True)

        ttk.Label(
            mainFrame,
            text='Enter amino acid sequences (one per line). Use "id,SEQUENCE" to set a custom id, '
                 'or just "SEQUENCE" for auto-numbering:',
        ).pack(anchor='w')

        self.inputText = tk.Text(mainFrame, height=8, wrap='word')
        self.inputText.pack(fill='x', pady=(4, 8))
        self.inputText.insert(
            '1.0',
            'example_1,GHFSTVWLCWDIQRKRFVALKVVKSAGHYTETAVDEIKLLKCVRDSDPSDPKRETIVQLIDDFRISGVNGVHVCMVLEVLGHQ\n'
            'example_2,RVINDDCPNLIGNRDLYKKVEWICEDCSNIFRNTGMATLCRKNCFFNEDFLWCVYATERTEEMSQLRQWVGILGAGRE\n',
        )

        buttonRow = ttk.Frame(mainFrame)
        buttonRow.pack(fill='x', pady=(0, 8))

        self.predictButton = ttk.Button(buttonRow, text='Predict', command=self.onPredictClicked)
        self.predictButton.pack(side='left')

        self.statusLabel = ttk.Label(buttonRow, text='Ready', foreground='#555555')
        self.statusLabel.pack(side='left', padx=12)

        self.progressBar = ttk.Progressbar(mainFrame, mode='indeterminate')
        self.progressBar.pack(fill='x', pady=(0, 8))

        ttk.Label(mainFrame, text='Results:').pack(anchor='w')

        columns = ('id', 'sequence', 'probability', 'prediction')
        self.resultTree = ttk.Treeview(mainFrame, columns=columns, show='headings', height=8)
        self.resultTree.heading('id', text='ID')
        self.resultTree.heading('sequence', text='Sequence')
        self.resultTree.heading('probability', text='Probability')
        self.resultTree.heading('prediction', text='Prediction')
        self.resultTree.column('id', width=90, anchor='w')
        self.resultTree.column('sequence', width=320, anchor='w')
        self.resultTree.column('probability', width=80, anchor='center')
        self.resultTree.column('prediction', width=200, anchor='center')
        self.resultTree.pack(fill='both', expand=True, pady=(4, 8))

        ttk.Label(mainFrame, text='Log / Error Messages:').pack(anchor='w')
        logFrame = ttk.Frame(mainFrame)
        logFrame.pack(fill='both', expand=True)
        self.logText = tk.Text(logFrame, height=8, wrap='word', state='disabled', bg='#f5f5f5')
        logScroll = ttk.Scrollbar(logFrame, command=self.logText.yview)
        self.logText.configure(yscrollcommand=logScroll.set)
        self.logText.pack(side='left', fill='both', expand=True)
        logScroll.pack(side='right', fill='y')

    def appendLog(self, text):
        self.logText.configure(state='normal')
        self.logText.insert('end', text + '\n')
        self.logText.see('end')
        self.logText.configure(state='disabled')

    def onPredictClicked(self):
        rawText = self.inputText.get('1.0', 'end')
        recordList = parseSequenceInput(rawText)
        if not recordList:
            messagebox.showwarning('No sequence', 'Please enter at least one amino acid sequence.')
            return

        self.predictButton.configure(state='disabled')
        self.statusLabel.configure(
            text='Predicting... (new sequences may take a few minutes, please wait)', foreground='#cc6600')
        self.progressBar.start(12)
        for row in self.resultTree.get_children():
            self.resultTree.delete(row)

        workerThread = threading.Thread(target=self.runPipeline, args=(recordList,), daemon=True)
        workerThread.start()
        self.root.after(200, self.pollQueue)

    def runPipeline(self, recordList):
        try:
            rows = runPredictionPipeline(
                recordList, GUI_INPUT_CSV, GUI_RESULT_CSV,
                onLog=lambda text: self.resultQueue.put(('log', text)))
            self.resultQueue.put(('done', rows))
        except Exception as exc:
            self.resultQueue.put(('error', f'Exception occurred: {exc}'))

    def pollQueue(self):
        try:
            while True:
                kind, payload = self.resultQueue.get_nowait()
                if kind == 'log':
                    self.appendLog(payload)
                elif kind == 'error':
                    self.appendLog(payload)
                    self.statusLabel.configure(text='Failed, see log below', foreground='#cc0000')
                    self.progressBar.stop()
                    self.predictButton.configure(state='normal')
                    return
                elif kind == 'done':
                    for row in payload:
                        prob = float(row['probability'])
                        self.resultTree.insert(
                            '', 'end',
                            values=(row['id'], row['sequence'], f'{prob:.4f}', row['prediction']))
                    self.appendLog('Prediction complete.')
                    self.statusLabel.configure(text='Done', foreground='#1a7f1a')
                    self.progressBar.stop()
                    self.predictButton.configure(state='normal')
                    return
        except queue.Empty:
            pass
        self.root.after(200, self.pollQueue)


def main():
    root = tk.Tk()
    NeuroPredictorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
