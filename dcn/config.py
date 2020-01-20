import os

ROOT = os.getcwd()


class GeneralConfig:
    """
    General configuration
    """

    def __init__(self):

        self.context_len = 600
        self.question_len = 30
        self.word_emb_path = os.path.join(
            ROOT, "dcn", "glove", "word_embeddings.npy")
        self.word2id_path = os.path.join(ROOT, "dcn", "glove", "word2id.pkl")
        self.id2word_path = os.path.join(ROOT, "dcn", "glove", "id2word.pkl")
        self.word_emb_text_path = os.path.join(
            ROOT, "dcn", "glove", "glove.6B.300d.txt")
        self.model_save_path = os.path.join(ROOT, "dcn", "checkpoints")
        self.data_dir_path = os.path.join(ROOT, "dcn", "data")
        self.train_context_data_path = os.path.join(
            ROOT, "dcn", "data", "train.context")
        self.train_question_data_path = os.path.join(
            ROOT, "dcn", "data", "train.question")
        self.train_answer_span_data_path = os.path.join(
            ROOT, "dcn", "data", "train.span")


class ModelConfig:
    """
    Model configuration
    """

    def __init__(self):

        self.hidden_dim = 200
        self.embedding_dim = 300

        # vector with zeros for unknown words
        self.decoding_steps = 4
        self.maxout_pool_size = 16

        self.lr = 0.001
        self.dropout_ratio = 0.15

        self.max_grad_norm = 5.0
        self.batch_size = 32
        self.num_epochs = 50

        self.print_every = 100
        self.save_every = 50000000
        self.eval_every = 1000
        # regularization constant
        self.reg_lambda = 0.00007
