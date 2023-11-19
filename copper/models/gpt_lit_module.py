import torch
from torch import nn
from torch.nn import functional as F

import lightning as L
from lightning import LightningModule, LightningDataModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from typing import Any, Optional
import tiktoken

cl100k_base = tiktoken.get_encoding("cl100k_base")

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)

class GPTLitModule(LightningModule):
    def __init__(
        self,
        GPT: torch.nn.Module,
        learning_rate=1e-3,
        n_embed=64,
        block_size=8,
        n_heads=4,
        drop_p=0,
        n_decoder_blocks=4,
        initial_lr = 1e-3,
        vocab_size_lit = 100277
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # ignoring net as the model weights themselves are not a hyperparam
        self.save_hyperparameters(logger=False, ignore=['model', 'GPT'])


        self.learning_rate = learning_rate
        self.initial_lr = initial_lr
        self.vocab_size_lit = vocab_size_lit
        # self.GPT = GPT
        # def __init__(self, vocab_size, block_size, n_embed, n_heads=4, n_blocks=4, drop_p=0): --> GPT Definition
        print("Inside GPTLIT, before GPT model")

        # Initialize GPT model

        # Define LightningModel
        self.model = GPT
        print("Inside GPTLIT, after GPT model")

        # Adding conditions to ensure the experiment runs


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)) == 0)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_parameters(self):
        return self.model.get_parameters()

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):        
        mask = self.mask if targets is not None else None
        return self.model(x, targets=targets, mask=mask)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits, loss = self.forward(x, targets=y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.initial_lr) # Use Initial LR

        return {"optimizer": optimizer}
