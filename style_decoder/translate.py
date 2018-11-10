from __future__ import division

from onmt.translate.translator import *

import torch
import argparse
import math
import codecs

parser = argparse.ArgumentParser(description='translate.py')


from onmt.opts import *

translate_opts(parser)


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = build_translator(opt, report_score=True)

    outF = codecs.open(opt.output, 'w', 'utf-8')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None
    for line in addone(codecs.open(opt.src, "r", "utf-8")):
        
        if line is not None:
            #srcTokens = line.split()
            srcBatch.append(line) #[srcTokens]
            if tgtF:
                tgtTokens = tgtF.readline().split() if tgtF else None
                tgtBatch += [tgtTokens]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        #predBatch, predScore, goldScore = translator.translate(srcBatch, tgtBatch)
        all_scores, all_predictions = translator.translate(src_data_iter=srcBatch, batch_size=2)
        predScore = all_scores
        predBatch = all_predictions
        #print(all_scores, all_predictions)
        
        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        if tgtF is not None:
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1
            outF.write(" ".join(predBatch[b][0]) + '\n')
            outF.flush()

            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if translator.tgt_dict.lower:
                    srcSent = srcSent.lower()
                print('SENT %d: %s' % (count, srcSent))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                print("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    tgtSent = ' '.join(tgtBatch[b])
                    if translator.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    print('GOLD %d: %s ' % (count, tgtSent))
                    print("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        print("[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][n])))

                print('')

        srcBatch, tgtBatch = [], []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if tgtF:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()


if __name__ == "__main__":
    main()
