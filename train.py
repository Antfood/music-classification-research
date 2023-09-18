from transformers import Wav2Vec2Processor
from torch.utils.data import DataLoader
from models.w2v2_model import Wav2Vec2Classifier


def train(
    model : Wav2Vec2Classifier,  
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

    print(f':: Training for: {epochs} epochs')

    for epoch in range(epochs):
        batch_loss = []

        for batch_idx, (X, y) in enumerate(dataloader):
            X = X.squeeze(dim=1)  # (B, C, T) -> (B, T)

            input_values = processor(
                X,
                return_tensors="pt",
                sampling_rate=sample_rate,
                pad=True,
                truncate=True,
            ).input_values

            input_values = input_values.squeeze(dim=0).to(model.device()) # (B, T) -> (T)
            y_hat = model(input_values)

            # is 0 and (num_classes - 1)
            y = y.to(model.device())
            
            loss = criterion(y_hat, y)

            batch_loss.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

            if batch_idx % 30 == 0:
                print(f"EPOCH: {epoch+1}   BATCH: {batch_idx}  BATCH LOSS: {loss.item()}")

        epoch_losses.append(sum(batch_loss) / len(dataloader))

    return epoch_losses

# Initialize your model, criterion and optimizer here
# model = Wav2Vec2Classification(num_classes=...)
# criterion = nn.CrossEntropyLoss()
# optim = torch.optim.AdamW(params=model.parameters(), lr=0.001)

# Then call train function
# train(model, dataloader, criterion, optim)
