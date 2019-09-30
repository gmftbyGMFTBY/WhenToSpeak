# WhenToTalk
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
7. [bert-as-service](https://github.com/hanxiao/bert-as-service)

## Dataset
Format:
1. Corpus folder have lots of sub folder, each named as the turn lengths of the conversations.
2. Each sub folder have lots of file which contains one conversation.
3. Each conversation file is the **tsv** format, each line have four element:
    * time
    * poster
    * reader
    * utterance

Create the dataset

```bash
# ubuntu / cornell, cf / ncf. Then the ubuntu-corpus folder will be created
# ubuntu-corpus have two sub folder (cf / ncf) for each mode
./data/run.sh ubuntu cf
```

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

Generate the graph of the context

```python
# generate the graph information of the train/test/dev dataset
./run.sh graph cornell when2talk 0
```

Generate the vocab of the dataset

```python
./run.sh vocab ubuntu 0 0
```

Train the model (seq2seq / seq2seq-cf / hred / hred-cf):

```python
# train the hred model on the 4th GPU
./run.sh train ubuntu hred 4
```

Translate the test dataset by applying the model

```python
# translate the test dataset by applying the hred model on 4th GPU
./run.sh translate ubuntu hred 4
```

Evaluate the result of the translated utterances

```python
# evaluate the translated result of the model on 4th GPU (BERTScore need it)
./run.sh eval ubuntu hred 4
```

## Experiment Result

1. Automatic evaluation

* Compare the PPL, BLEU4, Disctint-1, Distinct-2 score for all the models.
    
    Proposed classified methods need to be cascaded to calculate the BLEU4, BERTScore (the same format as the traditional models' results)
    
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
        <td>0.0405</td>
        <td>0.0223</td>
        <td>0.089</td>
        <td>0.3825</td>
      </tr>
      <tr>
        <td>hred</td>
        <td>0.04</td>
        <td>0.0219</td>
        <td>0.0867</td>
        <td>0.3758</td>
      </tr>
      <tr>
        <td>hred-cf</td>
        <td>0.0407</td>
        <td>0.0125</td>
        <td>0.0585</td>
        <td>0.3751</td>
      </tr>
      <tr>
        <td>proposed</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </table>


* F1 metric for measuring the accuracy for the timing of the speaking, only for classified methods (hred-cf, ...). The stat data shows that the number of the negative label is the half of the number of the positive label. **F1** and **Acc** maybe suitable for mearusing the result instead of the F1. In this settings, we care more about the precision in F1 metric.

2. Human judgments (engaging, ...)
    
    Invit the volunteer to chat with these models (seq2seq, hred, seq2seq-cf, hred-cf,) and score the models' performance accorading to the **Engaging**, **Fluent**, ...

3. Graph ablation learning

* F1 accuracy of predicting the speaking timing (hred-cf,)
* BLEU4, BERTScore, Distinct-1, Distinct-2

