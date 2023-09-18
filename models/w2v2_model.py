from transformers import Wav2Vec2ForCTC
from torch import nn

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes, freeze=True):
        super().__init__()

        print('Getting Wav2Vec2 base model...')
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base')

        if freeze:
            for param in self.wav2vec2.parameters(): #type: ignore
                param.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Linear(499 * 32, 256),  # based on out.logits.shape
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.out_activation = nn.Softmax(dim=1)  # softmax for classification

    def forward(self, x):
        x = self.wav2vec2(x).logits #type: ignore
        x = x.view(x.size(0), -1)
        x = self.classification_head(x)
        return self.out_activation(x)
    
    def device(self):
        return next(self.parameters()).device

