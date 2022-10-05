import argparse
import torch
import time
import json
import numpy as np
import math
import random
import xml.etree.ElementTree as ET
from subprocess import check_output

from model import Model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--model_dir', type=str, default="model/")
    parser.add_argument('--domain', type=str, default="laptop")

    args = parser.parse_args()
    return args

def append_helper(ix, tag_on, domain, sent, start, opins):
    end = ix
    tag_on = False
    if domain == 'restaurant':
        opin = ET.Element("Opinion")
        opin.attrib['target'] = sent.find('text').text[start:end]
    elif domain == 'laptop':
        opin = ET.Element("aspectTerm")
        opin.attrib['term'] = sent.find('text').text[start:end]
    opin.attrib['from'] = str(start)
    opin.attrib['to'] = str(end)
    opins.append(opin)
    return end, tag_on, opins

def label_xml(fn, output_fn, corpus, label, domain):
    dom = ET.parse(fn)
    root = dom.getroot()
    pred_y = []
    for zx, sent in enumerate(root.iter("sentence") ) :
        tokens = corpus[zx]
        lb = label[zx]
        if domain == 'restaurant':
            opins = ET.Element("Opinions")
        elif domain == 'laptop':
            opins = ET.Element("aspectTerms")
        token_idx, pt, tag_on=0, 0, False
        start, end = -1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx < len(tokens) and pt >= len(tokens[token_idx]):
                pt = 0
                token_idx+=1

            case_1 = token_idx<len(tokens) and lb[token_idx] == 1 and pt == 0 and c != ' '
            case_2 = token_idx<len(tokens) and lb[token_idx] == 2 and pt == 0 and c != ' ' and not tag_on
            case_3 = token_idx<len(tokens) and (lb[token_idx] == 0 or lb[token_idx] == 1) and tag_on and pt == 0
            case_4 = token_idx>=len(tokens) and tag_on

            if (case_1 and tag_on) or case_3 or case_4:
                end, tag_on, opins = append_helper(ix, tag_on, domain, sent, start, opins)
            
            if case_1 or case_2:
                start = ix
                tag_on = True
            
            quote_marks = ['``', "''"]
            if c == ' ':
                pass
            elif tokens[token_idx][pt:pt+2] in quote_marks:
                pt += 2
            else:
                pt += 1
        if tag_on:
            end, tag_on, opins = append_helper(len(sent.find('text').text), tag_on, domain, sent, start, opins)
        sent.append(opins)
    dom.write(output_fn)

def test(model, test_X, raw_X, domain, command, template, batch_size=128, crf=False):
    pred_y=np.zeros((test_X.shape[0], 83), np.int16)
    model.eval()
    for offset in range(0, test_X.shape[0], batch_size):
        batch_test_X_len=np.sum(test_X[offset:offset+batch_size]!=0, axis=1)
        batch_idx=batch_test_X_len.argsort()[::-1]
        batch_test_X_len=batch_test_X_len[batch_idx]
        batch_test_X_mask=(test_X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)
        batch_test_X=test_X[offset:offset+batch_size][batch_idx]
        batch_test_X_mask=torch.autograd.Variable(torch.from_numpy(batch_test_X_mask).long().cuda() )
        batch_test_X = torch.autograd.Variable(torch.from_numpy(batch_test_X).long().cuda() )
        batch_pred_y = model(batch_test_X_len, batch_test_X_mask, batch_test_X)
        r_idx=batch_idx.argsort()
        if crf:
            batch_pred_y=[batch_pred_y[idx] for idx in r_idx]
            for ix in range(len(batch_pred_y) ):
                for jx in range(len(batch_pred_y[ix]) ):
                    pred_y[offset+ix,jx]=batch_pred_y[ix][jx]
        else:
            batch_pred_y=batch_pred_y.data.cpu().numpy().argmax(axis=2)[r_idx]
            pred_y[offset:offset+batch_size,:batch_pred_y.shape[1]]=batch_pred_y
    model.train()
    assert len(pred_y)==len(test_X)
    
    command=command.split()
    if domain=='restaurant':
        label_xml(template, command[6], raw_X, pred_y, domain)
        acc=check_output(command ).split()
        print(acc)
        return float(acc[9][10:])
    elif domain=='laptop':
        label_xml(template, command[4], raw_X, pred_y, domain)
        acc=check_output(command ).split()
        print(acc)
        return float(acc[15])

def evaluate(runs, data_dir, model_dir, domain, command, template):
    ae_data=np.load(data_dir+domain+".npz")
    with open(data_dir+domain+"_raw_test.json") as f:
        raw_X=json.load(f)
    results=[]
    for r in range(runs):
        model=torch.load(model_dir+domain+str(r) )
        result=test(model, ae_data['test_X'], raw_X, domain, command, template, crf=False)
        results.append(result)
    print(sum(results)/len(results) )

if __name__ == "__main__":
    args = get_args()

    if args.domain=='restaurant':
        command="java -cp data/jar_data/A.jar absa16.Do Eval -prd data/official_data/pred.xml -gld data/official_data/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1"
        template="data/official_data/EN_REST_SB1_TEST.xml.A"
    elif args.domain=='laptop':
        command="java -cp data/jar_data/eval.jar Main.Aspects data/official_data/pred.xml data/official_data/Laptops_Test_Gold.xml"
        template="data/official_data/Laptops_Test_Data_PhaseA.xml"

    evaluate(args.runs, args.data_dir, args.model_dir, args.domain, command, template)
