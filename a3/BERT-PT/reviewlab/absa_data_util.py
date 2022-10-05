# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import torch
from torch.utils.data import TensorDataset


import logging
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample."""
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class AEProcessor(object):
    """Processor for Aspect Extraction."""

    def __init__(self, config):
        self.data_dir = config.data_dir

    def _load_data(self, fn, job_type):
        json_data = self._read_json(os.path.join(self.data_dir, fn))
        # format
        if "data" in json_data:
            json_data = json_data["data"]
        return self._create_examples(json_data, job_type)

    def get_train_examples(self, fn="train.json"):
        return self._load_data(fn, "train")

    def get_dev_examples(self, fn="dev.json"):
        return self._load_data(fn, "dev")

    def get_test_examples(self, fn="test.json"):
        return self._load_data(fn, "test")

    @classmethod
    def _read_json(cls, input_file):
        # Reads a json file for tasks in sentiment analysis.
        with open(input_file) as f:
            return json.load(f)

    def get_labels(self):
        return ["O", "B", "I"]

    def _create_examples(self, lines, job_type):
        # Creates examples for the training and dev sets.
        examples = []
        for ids in lines:
            guid = "%s-%s" % (job_type, ids)
            examples.append(
                InputExample(guid=guid, text_a=lines[ids]['sentence'], text_b=None, label=lines[ids]['label']) )
        return examples


class TokenCLSConverter(object):
    """Token Classification Converter (e.g., Aspect Extraction)."""

    @classmethod
    def convert_examples_to_features(cls, config,
                                     examples,
                                     tokenizer,
                                     max_seq_length,
                                     label_list,
                                     cls_token_at_end=False,
                                     cls_token="[CLS]",
                                     cls_token_segment_id=0,
                                     sep_token="[SEP]",
                                     pad_on_left=False,
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     pad_token_label_id=-100,
                                     sequence_a_segment_id=0,
                                     mask_padding_with_zero=True
                                     ):
        # map label BIO to id 012, pad is -100
        label_map = {"-100": pad_token_label_id}
        for (i, label) in enumerate(label_list):
            label_map.update({label: i})

        features = []
        for (idx, sentence) in enumerate(examples):
            if idx % 1e4 == 0:
                # print progress
                logger.info("Writing example %d / %d", idx, len(examples))

            tokens = []
            token_labels = []  # ids

            for word, label in zip(sentence.text_a, sentence.label):
                # tokenize word to maybe smaller tokens
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word,
                # and padding ids for the remaining tokens
                tmp = [pad_token_label_id] * (len(word_tokens))
                tmp[0] = label_map[label]
                token_labels.extend(tmp)

            sp_tokens_count = 2

            # cut
            if len(tokens) > max_seq_length - sp_tokens_count:
                tokens = tokens[:(max_seq_length - sp_tokens_count)]
                token_labels = token_labels[:(max_seq_length - sp_tokens_count)]

            # end of tokens
            tokens.append(sep_token)
            token_labels.append(pad_token_label_id)

            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # cls token
            if cls_token_at_end:
                tokens.append(cls_token)
                token_labels.append(pad_token_label_id)
                token_type_ids.append(cls_token_segment_id)
            else:
                # cls_token_at_start
                tokens = [cls_token] + tokens
                token_labels = [pad_token_label_id] + token_labels
                token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Pad sentence to max_seq_length
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            if mask_padding_with_zero:
                mask_padding = 0
                input_mask = [1] * len(input_ids)
            else:
                mask_padding = 1
                input_mask = [0] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)

            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([mask_padding] * padding_length) + input_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
                token_labels = ([pad_token_label_id] * padding_length) + token_labels
            else:
                # pad on right
                input_ids.extend([pad_token] * padding_length)
                input_mask.extend([mask_padding] * padding_length)
                token_type_ids.extend([pad_token_segment_id] * padding_length)
                token_labels.extend([pad_token_label_id] * padding_length)

            assert len(input_mask) == max_seq_length
            assert len(token_type_ids) == max_seq_length
            assert len(token_labels) == max_seq_length
            assert min(token_labels) >= -100 and max(token_labels) < len(label_list)

            if idx < 2:
                logger.info("*** Show Example ***")
                logger.info("guid: %s", sentence.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("label_ids: %s", " ".join([str(x) for x in token_labels]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("token_type_ids: %s", " ".join([str(x) for x in token_type_ids]))

            features.append(
                InputFeatures(input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, label=token_labels))

        return features


def build_dataset(features):
    # generate tensor and tensor dataset
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    label_list = []
    for feat in features:
        input_ids_list.append(feat.input_ids)
        attention_mask_list.append(feat.attention_mask)
        token_type_ids_list.append(feat.token_type_ids)
        label_list.append(feat.label)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids_list, dtype=torch.long)
    label = torch.tensor(label_list, dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, token_type_ids, label)