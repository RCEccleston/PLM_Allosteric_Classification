#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import pandas as pd
from Bio import pairwise2
from Bio.Seq import Seq
import ast

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print('Available device:', device)


def calculate_identity(alignment):
    """Calculate the sequence identity from a pairwise alignment."""
    seq1_aligned, seq2_aligned = alignment[0], alignment[1]
    matches = sum(res1 == res2 for res1, res2 in zip(seq1_aligned, seq2_aligned) if res1 != '-' and res2 != '-')
    identity = matches / min(len(seq1_aligned.replace('-', '')), len(seq2_aligned.replace('-', ''))) * 100
    return identity



# Function to filter sequences based on pairwise identity
def filter_sequences(df, N, threshold=30):
    """Filter out sequences that have >30% identity when aligned pairwise."""
    to_remove = set()
    to_add = set()
    # Convert the sequences in the DataFrame to Bio.Seq objects
    sequences = [Seq(seq) for seq in df['Sequences']]
    # Perform pairwise alignments and filter sequences
    for i in range(len(sequences)):
        print(i)
        for j in range(i + 1, len(sequences)):
#            print(i, j)
            # Perform pairwise alignment (global alignment using Needleman-Wunsch)
            alignments = pairwise2.align.globalxx(sequences[i], sequences[j])
            alignment = alignments[0]  # Get the best alignment

            # Calculate identity
            identity = calculate_identity(alignment)
            if identity > threshold:
                to_remove.add(j)  # Mark for removal if identity exceeds threshold
            if identity <= threshold:
                to_add.add(j)  # Add to test set index
                if len(to_add) == N:
                    break

    # Return DataFrame with filtered sequences
    test_set = df.iloc[list(to_add)]
    train_df = df.drop(to_add).reset_index(drop=True)
    return train_df, test_set



filename =  '/home/lshre1/Documents/PredAllo/processed_data_cleaned.csv'
print(filename)

df = pd.read_csv(filename)
df = df.dropna()
print(df.head(10))
sequences = df['Sequences']
labels = df['Labels'].apply(ast.literal_eval)

sequences = sequences.tolist()
labels = labels.tolist()
print(len(sequences))
total_N = len(sequences)
test_N = int(total_N * 0.1)
validation_N = int(total_N * 0.1)
train_N = total_N - test_N - validation_N
N = test_N + validation_N

print("calculating sequence identity")
# Apply the filter to the sequences
train_df, test_df = filter_sequences(df, N)


train_df.to_csv('passer_train_df.csv', index=False)
test_df.to_csv('passer_test_df.csv', index=False)







