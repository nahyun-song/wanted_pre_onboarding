### Transformer 아키텍처를 기반으로 하는 고성능 모델들

- GPT : Transformer 의 Decoder 아키텍처를 활용

- BERT : Transformer의 Encoder 아키텍처를 활용

​

### 모델의 변화

- Seq2Seq : 고정된 크기의 context vector를 사용하기 때문에 소스 문장들을 고정된 크기의 vector에 압축해야만 함

                     - > 병목이 발생하여 성능 하락읜 원인이 됨

- Attention : Seq2Seq의 단점을 보완 - context vector만을 참고하는 게 아니라 인코더의 모든 출력을 참고한 weighted sum vector를 구함

                     이후 모델들은 입력 시퀀스 전체에서 정보를 추출하는 방향으로 발전
                     
- Transformer : RNN, CNN 기반의 아키텍처를 사용하지 않고 Attention mechanism을 사용하여 성능을 향상  

## Transformer

- RNN이나 CNN 대신 Positional Encoding 사용

- RNN을 사용하지 않지만 인코더와 디코더로 구성

- Attention 과정을 여러 레이어에서 반복


### 입력 값 임베딩 (Enbedding)

 - RNN기반의 아키텍처는 순서에 맞게 들어감. 하지만 이를 사용하지 않는 경우에는 위치 정보를 포함하고 있는 임베딩을 사용해야 함

 - Positional Encoding : 각 단어의 상대적인 위치 정보를 담고 있는 인코딩. 주기 함수를 활용한 공식을 사용

 - 입력 문장에 대한 정보 + 위치에 대한 정보를 어텐션에 입력

​

### 인코더 (Encoder) 

- Self-Attention : 인코더 파트에서 수행하는 attention

                               각각의 단어끼리 어떤 연관성을 가지고 있는지 구하기 위해 사용 (attention score)

                               -> 문맥에 대한 정보를 잘 학습하기 위한 장치

- Residual Learning : 특정 레이어를 건너 뛰어서 입력. 기존 정보는 입력 받으면서 추가적으로 잔여부분만 학습

                                    학습 난이도 낮아짐 -> 속도 빨라짐 -> global optima를 찾을 확률 높아짐 -> 성능 향상

- Normalization


### The Transformer - model architecture. 

- 한 레이어에서 Attention과 Normalization 과정 반복

- 여러 개의 레이어를 중첩해서 사용

- 각각의 레이어는 서로 다른 파라미터를 가짐

​

### 인코더와 디코더 (Encoder and Decoder)

- 단어 정보 + 위치 정보를 입력

- 한 레이어에 두 개의 Attention이 존재

    - Self-Attention : 인코더 파트의 self-attention과 동일

    - Encoder-Decoder Attention : 디코더에 입력된 출력 단어가 입력 단어 중에서 어떤 정보와 높은 연관성을 가지는지 계산

                                                            인코더의 마지막 레이어에서 나온 출력값을 입력

- 각각의 레이어를 중첩해서 사용

- 마지막 인코더 레이어의 출력은 모든 디코더 레이어에 각각 입력됨 

​

### 어텐션 (Attention)


- Multi-Head Attention

    - 입력

        - Query : 무언가를 물어보는 주체

        - Key : 물어보는 대상

        - Value

    - Scaled Dot-Product Attention

        - Attention score : 입력된 Query와 각각의 Key 간의 연관성을 행렬곱으로 구함

        - Attention value : Attention score와 Value의 곱

    - h개의 attention concept을 학습

    - 입력과 출력의 dimension을 일치시키기 위해 h개의 attention을 concat

    - 전체 아키텍처에서 동일한 함수로 동작하지만 위치마다 Q, K, V를 어떻게 사용할지는 달라질 수 있음

​

### 어텐션(Attention)의 종류

- Encoder Self-Attention : 각각의 단어가 서로 어떤 연관성을 가지는지 어텐션은 통해 구함            

                                                전체 문장에 대한 representation을 learning할 수 있도록 함

- Masked Decoder Self-Attention : 각각의 출력 단어가 모든 단어를 참고하는 것이 아니라 앞쪽에 등장했던 출력 단어들을 참고

                                                                뒤쪽 단어를 참고하면 cheating을 할 수 있기 때문

- Encoder-Decoder Attention : Query는 Decoder에 있고 각각의 Key와 Value는 Encoder에 있는 상황

                                                         Query가 Key와 Value를 참고

​

### BLEU score (BiLingual Evaluation Understudy)

 - 한 자연어에서 다른 자연어로 기계 번역된 텍스트의 품질을 평가하기 위한 알고리즘

 - 후보 텍스트가 참조 텍스트와 얼마나 유사한지를 0과 1사이의 값으로 나타냄 (1에 가까울수록 유사함)

 - N-gram phrase가 얼마나 겹치는지 반영

 - Recall은 사용하지 않고 Precision만을 사용


- 1,2,3,4 -gram 의 Precision을 사용한 뒤 기하평균을 적용

- 짧은 문장일 때 precision 값이 높아질 확률이 높으므로 gravity penalty 부여

​

​

​

​

💡 참고

- 논문 Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf

- 딥러닝 논문 리뷰  https://youtu.be/AA621UofTUA

- BLEU score 관련 https://en.wikipedia.org/wiki/BLEU, https://sundryy.tistory.com/81