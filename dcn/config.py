import os


class Config:
    """
    """

    def __init__(self):
        self.context_len = 600
        self.question_len = 30

        self.hidden_dim = 200
        self.embedding_dim = 100

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
