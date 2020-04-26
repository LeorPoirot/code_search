import torch.nn as nn


class CodeRetrievalModel(nn.Module):
    def __init__(self, code_encoder, comment_encoder, opt):
        super(CodeRetrievalModel, self).__init__()
        self.code_encoder = code_encoder
        self.comment_encoder = comment_encoder
        self.opt = opt

    def forward(self, code_batch, comment_batch, bad_comment_batch):
        code_feat = self.code_encoder(code_batch)
        good_comment_feat = self.comment_encoder(comment_batch)
        bad_comment_feat = self.comment_encoder(bad_comment_batch)

        return code_feat, good_comment_feat, bad_comment_feat
