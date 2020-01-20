import torch.LongTensor as LT
from torch.utils.data import Dataset
from dcn.config import GeneralConfig
import pickle
from nltk import word_tokenize
config = GeneralConfig()

_UNK = b"<unk>"


class SquadDataset(Dataset):
    """
    Squad dataset
    """

    def __init__(self):
        """
        constructor function
        """
        super(SquadDataset, self).__init__()
        # load word2id dictionary
        with open(config.word2id_path) as f:
            self.word2id = pickle.load(f)
        # load context data
        with open(config.train_context_data_path) as f:
            self.context_data = f.readlines()
        # load question data
        with open(config.train_question_data_path) as f:
            self.quesiton_data = f.readlines()
        # load answer span data
        with open(config.train_answer_span_data_path) as f:
            self.answer_span_data = f.readlines()

    def __len__(self):
        return len(self.answer_span_data)

    def _padding(self, token_ids, max_len):

        sent_len = len(token_ids)
        if sent_len > max_len:
            return token_ids[:max_len]

        token_ids = token_ids + (max_len-sent_len)*[0]
        return token_ids

    def sentence_tokenids(self, sentence, max_len):
        """
        Returns token ids of the sentence
        """
        token_ids = [self.word2id.get(word, _UNK)
                     for word in word_tokenize(sentence)]
        padded_token_ids = self._padding(token_ids, max_len=max_len)

        return padded_token_ids, len(token_ids)

    def __getitem__(self, index):
        """
        """
        context = self.context_data[index]
        context_ids, context_len = self.sentence_tokenids(
            context, max_len=config.context_len)
        question = self.quesiton_data[index]
        question_ids, question_len = self.sentence_tokenids(
            question, max_len=config.question_len)
        answer_span = self.answer_span_data[index]

        return (LT(context_ids), LT([context_len]), LT(question_ids), LT([question_len]), LT(answer_span))
