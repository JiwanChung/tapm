#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading
import inspect
from pathlib import Path


# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class VistMeteor:

    def __init__(self):
        curr_path = Path(inspect.getfile(self.__class__)).parent
        meteor_path = str(curr_path / METEOR_JAR)
        # Here is MSR VIST group's measurement
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', meteor_path, '-', '-', '-stdio', '-t', 'hter']
        # self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
        #         '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=os.path.dirname(os.path.abspath(__file__)), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        print("getting stat")

        lengths = []
        for i in imgIds:
            assert(len(res[i]) == 1)
            lengths.append(len(gts[i]))
            for gt in gts[i]:
                stat = self._stat(res[i][0], [gt])
                eval_line += ' ||| {}'.format(stat)

        print("evaluating")
        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        self.meteor_p.stdin.flush()

        for img in lengths:
            img_score = []
            for i in range(img):
                img_score.append(float(self.meteor_p.stdout.readline().strip()))
            img_score = max(img_score)
            scores.append(img_score)
        # score = float(self.meteor_p.stdout.readline().strip())
        score = sum(scores) / len(scores)
        self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
        self.meteor_p.stdin.flush()

        return self.meteor_p.stdout.readline().decode().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()
