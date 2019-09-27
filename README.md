# WhenToSpeak
Make the model decide when to utter the utterances in the conversation, which can make the interaction more engaging.

Model architecture:
1. GCN for predicting the timing of speaking
    * Dialogue-sequence: Sequence of the dialogue history
    * User-sequence: User utterance sequence
    * PMI: Context relationship
2. (Seq2Seq/HRED) for language generation
3. Multi-head attention for dialogue context (use GCN hidden state)

## Requirements
1. Pytorch 1.2
2. [PyG](https://github.com/rusty1s/pytorch_geometric)
3. numpy
4. tqdm
5. [BERTScore](https://github.com/Tiiiger/bert_score)
6. nltk: word tokenize

## Dataset
Ubuntu Dialogue v1, format:


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

Generate the vocab of the dataset

```python
./run.sh vocab 0 0
```

Train the model (seq2seq / seq2seq-cf / hred / hred-cf):

```python
# train the hred model on the 4th GPU
./run.sh train hred 4
```

Translate the test dataset by applying the model

```python
# translate the test dataset by applying the hred model on 4th GPU
./run.sh translate hred 4
```

Evaluate the result of the translated utterances

```python
# evaluate the translated result of the model on 4th GPU (BERTScore need it)
./run.sh eval hred 4
```

## Experiment Result

1. Automatic evaluation

* Compare the BLEU4, BERTScore, Disctint-1, Distinct-2 score for all the models.
    
    Proposed classified methods need to be cascaded to calculate the BLEU4, BERTScore (the same format as the traditional models' result)
    
    <table align="center">
      <tr>
        <th>Models</th>
        <th>BLEU4</th>
        <th>Dist-1</th>
        <th>Dist-2</th>
        <th>BERTScore</th>
      </tr>
      <tr>
        <td>seq2seq</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>hred</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>hred-cf</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>proposed</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </table>


* F1 metric for measuring the accuracy for the timing of the speaking, only for classified methods (hred-cf, ...)

2. Human judgments (engaging, ...)
    
    Invit the volunteer to chat with these models (seq2seq, hred, seq2seq-cf, hred-cf,) and score the models' performance accorading to the **Engaging**, **Fluent**, ...

3. Graph ablation learning

* F1 accuracy of predicting the speaking timing (hred-cf,)
* BLEU4, BERTScore, Distinct-1, Distinct-2

