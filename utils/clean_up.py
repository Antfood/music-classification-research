import torchaudio
import pandas as pd

df = pd.read_csv("data/split_audio.csv")
drop_indices = []

for index, row in df.iterrows():
    audio_path = row['audio_path']
    audio, sr = torchaudio.load(audio_path)  # type: ignore

    if audio.shape[1] == 0:
        print(f"Audio '{audio_path}' has no data. shape: {audio.shape}. Dropping row.")
        drop_indices.append(index)

# Drop rows with empty audio files
df.drop(drop_indices, inplace=True)

# Save the updated DataFrame to a new CSV file
df.to_csv("data/cleaned_split_audio.csv", index=False)

