import torch
from torch.utils.data import DataLoader
from dcn.dataloader import SquadDataset
from dcn.model import DCN
from dcn.config import GeneralConfig, ModelConfig
from tqdm import tqdm, trange
import os
import numpy as np

use_cuda = True if torch.cuda.is_available() else False


if __name__ == "__main__":

    mconfig = ModelConfig()
    gconfig = GeneralConfig()
    weights = torch.FloatTensor(np.load(gconfig.word_embedding_path))

    model = DCN(weight=weights)
    if use_cuda:
        model = model.cuda()
    #=============== Define dataloader ================#
    train_dataset = SquadDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=mconfig.batch_size)
    #============== Define optimizer ================#
    opt = torch.optim.Adam(model.parameters(), lr=mconfig.lr)

    print("Training started!")
    for epoch in trange(mconfig.epochs, desc="Epoch"):

        for iteration, batch in enumerate(tqdm(train_dataloader)):
            # unpack the batch
            context, context_lens, question, question_lens, answer_spans = batch

            if use_cuda:
                context = context.cuda()
                context_lens = context_lens.cuda()
                question = question.cuda()
                question_lens = question_lens.cuda()
                answer_spans = answer_spans.cuda()
            # Zero out the gradients before forward pass
            opt.zero_grad()
            loss = model(context, context_lens.view(-1), question,
                         question_lens.view(-1), answer_spans)
            loss.backward()
            opt.step()
            total_loss = loss.item()

            if (iteration+1) % mconfig.print_frequency == 0:
                print(
                    f"Epoch: {epoch+1} Iteration: {iteration+1} loss: {total_loss}")

        print("Saving states")
        #================ Saving states ==========================#
        if not os.path.exists(gconfig.model_save_path):
            os.mkdir(gconfig.model_save_path)
        # save model state
        torch.save(model.state_dict(), gconfig.model_save_path +
                   f'/model_epoch_{epoch+1}.pt')
    print("Training completed!!!")
