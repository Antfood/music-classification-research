from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from torch import nn
from torch.utils.data import DataLoader

class Wav2Vec2Regression(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

        print('Getting Wav2Vec2 base model...')
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base')

        if freeze:
            for param in self.wav2vec2.parameters(): #type: ignore
                param.requires_grad = False

        self.regression_head = nn.Sequential(
            nn.Linear(499 * 32, 256),  # based on out.logits.shape
            nn.Linear(256, 1)
        )

        self.out_activation = nn.Sigmoid() # sigmoid for regression

    def forward(self, x):
        x = self.wav2vec2(x).logits #type: ignore
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)
        return self.out_activation(x)
    
    def device(self):
        return next(self.parameters()).device


def train(
    model: Wav2Vec2Regression,
    dataloader: DataLoader,
    criterion,
    optim,
    sample_rate: int = 16000,
    epochs: int = 5,
) -> list[float]:


    print('Getting Wav2Vec2 Processor...')
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model.train()
    epoch_losses = []

    print(f':: Taining for: {epochs} epochs')

    for epoch in range(epochs):
        batch_loss = []

        for batch_idx, (X, y) in enumerate(dataloader):
            X = X.squeeze(dim=1)  # remove channel dimension

            input_values = processor(
                X,
                return_tensors="pt",
                sampling_rate=sample_rate,
                pad=True,
                truncate=True,
            ).input_values

            input_values = input_values.squeeze(dim=0).to(model.device())
            y_hat = model(input_values)

            y_norm = y / 100.0 # max value is 100
            y_norm = y_norm.to(model.device())
            loss = criterion(y_hat.squeeze(dim=1), y_norm.to(dtype=torch.float32))

            batch_loss.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

            if batch_idx % 100 == 0:
                print( f"EPOCH: {epoch+1}   BATCH: {batch_idx}  BATCH LOSS: {loss.item()}")

        epoch_losses.append(sum(batch_loss) / len(dataloader))

    return epoch_losses
