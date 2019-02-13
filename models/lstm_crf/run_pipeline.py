# -*- coding:utf-8 -*-
"""
file name: run_pipeline.py
Created on 2019/2/12
@author: kyy_b
@desc:
"""
import subprocess
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python main.py opinion_id")
        sys.exit(0)
    opinion_id = sys.argv[1]

    cmd = 'python ../../data/example/build_vocab.py {}'.format(opinion_id)
    print(cmd)
    subprocess.call(cmd, shell=True)

    cmd = 'python ../../data/example/build_w2v.py {}'.format(opinion_id)
    print(cmd)
    subprocess.call(cmd, shell=True)

    time_start = time.time()
    cmd = 'python main.py {}'.format(opinion_id)
    print(cmd)
    subprocess.call(cmd, shell=True)
    time_end = time.time()
    print("train model {} spend {}s".format(opinion_id, time_end - time_start))

    cmd = 'python export.py {}'.format(opinion_id)
    print(cmd)
    subprocess.call(cmd, shell=True)

    print("entity_level_score")
    cmd = "perl ../conlleval < /data/kongyy/nlp/tf_ner_guillaumegenthial/results_{}/score/train.preds.txt > /data/kongyy/nlp/tf_ner_guillaumegenthial/results_{}/score/entity_level_score.txt".format(opinion_id, opinion_id)
    subprocess.call(cmd, shell=True)
    cmd = "perl ../conlleval < /data/kongyy/nlp/tf_ner_guillaumegenthial/results_{}/score/test.preds.txt >> /data/kongyy/nlp/tf_ner_guillaumegenthial/results_{}/score/entity_level_score.txt".format(opinion_id, opinion_id)
    subprocess.call(cmd, shell=True)

    # cmd = "python predict /data/kongyy/nlp/tf_ner_guillaumegenthial/xuanjia.words.txt"
    # print(cmd)
    # subprocess.call(cmd, shell=True)

    # cmd = 'python serve.py {}'.format(opinion_id)
    # print(cmd)
    # subprocess.call(cmd, shell=True)
