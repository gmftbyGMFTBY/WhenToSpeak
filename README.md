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
5. nltk: word tokenize and sent tokenize

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
1. Language Model: BLEU4, PPL, Distinct-1, Distinct-2
2. Talk timing: F1, Acc
3. Human Evaluation: Engaging evaluation

## Baselines
### 1. Traditional methods

1. Seq2Seq
2. HRED / HRED + CF

### 2. Graph ablation learning
1. w/o BERT Embedding cosine similarity
2. w/o User-sequence
3. w/o Dialogue-sequence

## How to use

Generate the graph of the context

```python
# generate the graph information of the train/test/dev dataset
# The average context coverage in the graph: 0.7935/0.7949/0.7794 (train/test/dev) dataset
./run.sh graph cornell when2talk 0
```

Analyze the graph context coverage information

```python
./run.sh stat cornell 0 0
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

1. Methods
    * Seq2Seq: seq2seq with attention
    * HRED: hierarchical context modeling
    * HRED-CF: HRED model with classification for talk timing
    * When2Talk: GCNContext modeling first and RNN Context later
    * W2T_RNN_First: **Bi**RNN Context modeling first and GCNContext later
    * GCNRNN: combine the GCNContext and RNNContext together
    * GatedGCN: combine the Gated GCNContext and RNNContext together
    * W2T_GCNRNN: RNN + GCN combine RNN together (W2_T_RNN_First + GCNRNN)

2. Automatic evaluation

    * Compare the PPL, BLEU4, Disctint-1, Distinct-2 score for all the models.
    
        Proposed classified methods need to be cascaded to calculate the BLEU4, BERTScore (the same format as the traditional models' results)
    
    <table align="center">
      <tr>
        <th align="center" rowspan="2">Model</th>
        <th align="center" colspan="4">Dailydialog</th>
        <th align="center" colspan="4">Cornell</th>
      </tr>
      <tr>
        <td align="center">BLEU</td>
        <td align="center">Dist-1</td>
        <td align="center">Dist-2</td>
        <td align="center">PPL</td>
        <td align="center">BLEU</td>
        <td align="center">Dist-1</td>
        <td align="center">Dist-2</td>
        <td align="center">PPL</td>
      </tr>
      <tr>
        <td align="center">Seq2Seq</td>
        <td>0.1038</td>
        <td>0.0178</td>
        <td>0.072</td>
        <td>29.0640</td>
        <td>0.0843</td>
        <td>0.0052</td>
        <td>0.0164</td>
        <td>45.1504</td>
      </tr>
      <tr>
        <td align="center">HRED</td>
        <td>0.1175</td>
        <td>0.0176</td>
        <td>0.0576</td>
        <td>29.7402</td>
        <td>0.0823</td>
        <td>0.0227</td>
        <td>0.0524</td>
        <td>39.9009</td>
      </tr>
      <tr>
        <td align="center">HRED-CF</td>
        <td>0.1276</td>
        <td>0.0274</td>
        <td>0.0817</td>
        <td>22.4121</td>
        <td>0.1116</td>
        <td>0.0094</td>
        <td>0.0228</td>
        <td>38.2598</td>
      </tr>
      <tr>
        <td align="center">When2Talk</td>
        <td>0.1226</td>
        <td>0.0211</td>
        <td>0.0608</td>
        <td>24.0131</td>
        <td>0.0996</td>
        <td>0.0036</td>
        <td>0.0073</td>
        <td>32.9503</td>
      </tr>
      <tr>
        <td align="center">W2T_RNN_First</td>
        <td>0.1250</td>
        <td>0.0185</td>
        <td>0.0507</td>
        <td>25.3581</td>
        <td>0.1099</td>
        <td>0.007</td>
        <td>0.0172</td>
        <td>35.6625</td>
      </tr>
      <tr>
        <td align="center">GCNRNN</td>
        <td>0.1250</td>
        <td>0.0214</td>
        <td>0.0624</td>
        <td>23.9867</td>
        <td>0.1072</td>
        <td>0.0077</td>
        <td>0.0188</td>
        <td>33.9572</td>
      </tr>
      <tr>
        <td align="center">W2T_GCNRNN</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td align="center">GatedGCN</td>
        <td>0.1258</td>
        <td>0.0238</td>
        <td>0.0685</td>
        <td>25.0131</td>
        <td>0.1127</td>
        <td>0.0062</td>
        <td>0.0149</td>
        <td>34.2847</td>
      </tr>
    </table>

    * F1 metric for measuring the accuracy for the timing of the speaking, only for classified methods (hred-cf, ...). The stat data shows that the number of the negative label is the half of the number of the positive label. **F1** and **Acc** maybe suitable for mearusing the result instead of the F1. In this settings, we care more about the precision in F1 metric.

    <table align="center">
      <tr>
        <th align="center" rowspan="2">Model</th>
        <th align="center" colspan="2">Dailydialog</th>
        <th align="center" colspan="2">Cornell</th>
      </tr>
      <tr>
        <td align="center">Acc</td>
        <td align="center">F1</td>
        <td align="center">Acc</td>
        <td align="center">F1</td>
      </tr>
      <tr>
        <td>HRED-CF</td>
        <td>0.8222</td>
        <td>0.8645</td>
        <td>0.7708</td>
        <td>0.8427</td>
      </tr>
      <tr>
        <td>When2Talk</td>
        <td>0.7992</td>
        <td>0.8507</td>
        <td>0.7616</td>
        <td>0.8388</td>
      </tr>
      <tr>
        <td>W2T_RNN_First</td>
        <td>0.7522</td>
        <td>0.8358</td>
        <td>0.7323</td>
        <td>0.8322</td>
      </tr>
      <tr>
        <td>GCNRNN</td>
        <td>0.8176</td>
        <td>0.8635</td>
        <td>0.7598</td>
        <td>0.8445</td>
      </tr>
      <tr>
        <td>W2T_GCNRNN</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GatedGCN</td>
        <td>0.8016</td>
        <td>0.8526</td>
        <td>0.7594</td>
        <td>0.8445</td>
      </tr>
    </table>


2. Human judgments (engaging, ...)
    
    Invit the volunteer to chat with these models (seq2seq, hred, seq2seq-cf, hred-cf,) and score the models' performance accorading to the **Engaging**, **Fluent**, ...
    
    * Dailydialog dataset
        <table>
          <tr>
            <th align="center" rowspan="2">Model</th>
            <th align="center" colspan="3">When2Talk vs.</th>
            <th rowspan="2">kappa</th>
          </tr>
          <tr>
            <td>win(%)</td>
            <td>loss(%)</td>
            <td>tie(%)</td>
          </tr>
          <tr>
            <td>Seq2Seq</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
          </tr>
          <tr>
            <td>HRED</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
          </tr>
          <tr>
            <td>HRED-CF</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
          </tr>
        </table>
        
    * Cornell dataset
        <table>
          <tr>
            <th align="center" rowspan="2">Model</th>
            <th align="center" colspan="3">When2Talk vs.</th>
            <th rowspan="2">kappa</th>
          </tr>
          <tr>
            <td>win(%)</td>
            <td>loss(%)</td>
            <td>tie(%)</td>
          </tr>
          <tr>
            <td>Seq2Seq</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
          </tr>
          <tr>
            <td>HRED</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
          </tr>
          <tr>
            <td>HRED-CF</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
          </tr>
        </table>

3. Graph ablation learning
    
    * F1 accuracy of predicting the speaking timing (hred-cf,)
    * BLEU4, BERTScore, Distinct-1, Distinct-2

