# WhenToSpeak
Make the model decide when to utter the utterances in the conversation, which can make the interaction more engaging.

Model architecture:
1. GCN for predicting the timing of speaking
    * Dialogue-sequence: Sequence of the dialogue history
    * User-sequence: User utterance sequence
    * PMI: Context relationship
2. (Seq2Seq/HRED) for language generation
3. Multi-head attention for dialogue context (use GCN hidden state)

## Dataset
Ubuntu Dialogue v1

## Metric
1. F1: timing of speaking
2. BERTScore: Utterance quality
3. Human Evaluation: Engaging evaluation

## Baselines
### 1. Traditional methods

1. Seq2Seq / Seq2Seq + CF
2. HRED / HRED + CF

### 2. Graph ablation learning
1. w/o PMI
2. w/o User-sequence
3. w/o Dialogue-sequence

## How to use


## Experiment Result

