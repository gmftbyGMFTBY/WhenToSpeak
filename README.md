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
./run.sh graph cornell when2talk 0
```

Analyze the graph context coverage information

```python
# The average context coverage in the graph: 0.7935/0.7949/0.7794 (train/test/dev) dataset
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

Generate performance curve

```python
./run.sh curve dailydialog hred-cf 0
```

## Experiment Result

```markdown
wait to do:
1. add GatedGCN to all the graph-based method
2. add BiGRU to all the graph-based method
3. refer the DialogueGCN to construct the graph
    * the complete graph in the **p** windows size
    * add one long edge out of the windows size to explore long context sentence
    * user embedding as the node for processing
4. Layers analyse of the GatedGCN in this repo and mutli-turn modeling
```

1. Methods
    * Seq2Seq: seq2seq with attention
    * HRED: hierarchical context modeling
    * HRED-CF: HRED model with classification for talk timing
    * When2Talk: GCNContext modeling first and RNN Context later
    * W2T_RNN_First: **Bi**RNN Context modeling first and GCNContext later
    * GCNRNN: combine the Gated GCNContext and RNNContext together (?)
    * GatedGCN: combine the Gated GCNContext and RNNContext together
        1. BiRNN for background modeling
        2. Gated GCN for context modeling
        2. Combine GCN embedding and BiRNN embedding, final embedding
        4. Low-turn examples trained without the GCNConv (only use the BiRNN)
        5. Separate the decision module and generation module is better
    * W2T_GCNRNN: RNN + GCN combine RNN together (W2T_RNN_First + GCNRNN)

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
            <td>0.0571</td>
            <td>29.7402</td>
            <td>0.0823</td>
            <td>0.0227</td>
            <td>0.0524</td>
            <td>39.9009</td>
          </tr>
          <tr>
            <td align="center">HRED-CF</td>
            <td>0.1268</td>
            <td>0.0435</td>
            <td>0.1567</td>
            <td>29.0111</td>
            <td>0.1132</td>
            <td>0.0221</td>
            <td>0.0691</td>
            <td>38.5633</td>
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
            <td>0.1244</td>
            <td>0.0268</td>
            <td>0.0787</td>
            <td>24.5056</td>
            <td>0.1118</td>
            <td>0.0065</td>
            <td>0.0147</td>
            <td>33.754</td>
          </tr>
          <tr>
            <td align="center">GCNRNN</td>
            <td>0.1250</td>
            <td>0.0214</td>
            <td>0.0624</td>
            <td>25.8213</td>
            <td>0.1072</td>
            <td>0.0077</td>
            <td>0.0188</td>
            <td>33.9572</td>
          </tr>
          <tr>
            <td align="center">W2T_GCNRNN</td>
            <td>0.1246</td>
            <td>0.0152</td>
            <td>0.0400</td>
            <td>23.4434</td>
            <td>0.1107</td>
            <td>0.0063</td>
            <td>0.0142</td>
            <td>34.4256</td>
          </tr>
          <tr>
            <td align="center">GatedGCN</td>
            <td>0.1231</td>
            <td>0.0423</td>
            <td>0.1609</td>
            <td>27.1615</td>
            <td>0.1157</td>
            <td>0.0261</td>
            <td>0.0873</td>
            <td>34.4256</td>
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
            <td>0.8272</td>
            <td>0.8666</td>
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
            <td>0.8144</td>
            <td>0.8584</td>
            <td>0.7481</td>
            <td>0.8312</td>
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
            <td>0.7565</td>
            <td>0.8434</td>
            <td>0.7853</td>
            <td>0.8466</td>
          </tr>
          <tr>
            <td>GatedGCN</td>
            <td>0.8226</td>
            <td>0.8663</td>
            <td>0.738</td>
            <td>0.8181</td>
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

