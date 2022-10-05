import xml.etree.ElementTree as ET
from subprocess import check_output

import logging
logger = logging.getLogger(__name__)

        
class AEMetric(object):
    # command to run official eval.jar to get f1 score
    commands = {
        "rest": "./java -cp reviewlab/eval/A.jar absa16.Do Eval -prd data/ft/ae/16/rest/rest_pred.xml -gld data/ft/ae/16/rest/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1",
        "laptop": "./java -cp reviewlab/eval/eval.jar Main.Aspects data/ft/ae/14/laptop/laptop_pred.xml data/ft/ae/14/laptop/Laptops_Test_Gold.xml"

    }
    
    templates = {
        "rest": "data/ft/ae/16/rest/EN_REST_SB1_TEST.xml.A",
        "laptop": "data/ft/ae/14/laptop/Laptops_Test_Data_PhaseA.xml"
    }
    
    @classmethod
    def metric_name(cls, index):
        if index == 0:
            return "f1"
        else:
            raise Exception("unknown index")

    @classmethod
    def append_helper(cls, ix, tag_on, domain, sent, start, opins):
        end = ix
        tag_on = False
        if domain == 'rest':
            opin = ET.Element("Opinion")
            opin.attrib['target'] = sent.find('text').text[start:end]
        elif domain == 'laptop':
            opin = ET.Element("aspectTerm")
            opin.attrib['term'] = sent.find('text').text[start:end]
        opin.attrib['from'] = str(start)
        opin.attrib['to'] = str(end)
        opins.append(opin)
        return end, tag_on, opins

    @classmethod
    def label_xml(cls, fn, output_fn, corpus, label, domain):
        dom = ET.parse(fn)
        root = dom.getroot()
        pred_y = []
        for zx, sent in enumerate(root.iter("sentence")):
            tokens = corpus[zx]
            lb = label[zx]
            if domain == 'rest':
                opins = ET.Element("Opinions")
            elif domain == 'laptop':
                opins = ET.Element("aspectTerms")
            token_idx, pt, tag_on = 0, 0, False
            start, end = -1, -1
            for ix, c in enumerate(sent.find('text').text):
                if token_idx < len(tokens) and pt >= len(tokens[token_idx]):
                    pt = 0
                    token_idx += 1

                case_1 = token_idx < len(tokens) and lb[token_idx] == 1 and pt == 0 and c != ' '
                case_2 = token_idx < len(tokens) and lb[token_idx] == 2 and pt == 0 and c != ' ' and not tag_on
                case_3 = token_idx < len(tokens) and (lb[token_idx] == 0 or lb[token_idx] == 1) and tag_on and pt == 0
                case_4 = token_idx >= len(tokens) and tag_on

                if (case_1 and tag_on) or case_3 or case_4:
                    end, tag_on, opins = AEMetric.append_helper(ix, tag_on, domain, sent, start, opins)

                if case_1 or case_2:
                    start = ix
                    tag_on = True

                quote_marks = ['``', "''"]
                if c == ' ':
                    pass
                elif tokens[token_idx][pt:pt + 2] in quote_marks:
                    pt += 2
                else:
                    pt += 1
            if tag_on:
                end, tag_on, opins = AEMetric.append_helper(len(sent.find('text').text), tag_on, domain, sent, start, opins)
            sent.append(opins)
        dom.write(output_fn)

    @classmethod
    def evaluate(cls, config, pred_json):
        y_pred = [[pred_json["labels"].index(pred) for pred in preds] for preds in pred_json["preds_list"]]

        command = AEMetric.commands[config.domain].split()
        if config.domain == "rest":
            AEMetric.label_xml(AEMetric.templates[config.domain], command[6], pred_json["sentence"], y_pred, config.domain)
            result = check_output(command).split()
            logger.info("**** java output ****")
            logger.info("%s", result)
            return [float(result[9][10:])]
        elif config.domain == "laptop":
            AEMetric.label_xml(AEMetric.templates[config.domain], command[4], pred_json["sentence"], y_pred, config.domain)
            result = check_output(command).split()
            logger.info("**** java output ****")
            logger.info("%s", result)
            return [float(result[15])]
        else:
            raise ValueError("unknown domain %s.", config.domain)
