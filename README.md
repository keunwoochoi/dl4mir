# dl4mir: A Tutorial on Deep Learning for MIR

by [Keunwoo Choi](https://keunwoochoi.wordpress.com) (first.last@qmul.ac.uk)

This is a repo for my tutorial paper; [A Tutorial on Deep Learning for Music Information Retrieval](https://arxiv.org/abs/1709.04396). 

## Tutorials
  1. [Example 1: Pitch detector with a dense layer](https://github.com/keunwoochoi/dl4mir/blob/master/Example%201%20-%20a%20pitch%20detection%20network%20with%20Dense%20layers.ipynb)
  2. [Example 2: Chord recogniser with a convnet](https://github.com/keunwoochoi/dl4mir/blob/master/Example%202%20-%20a%20chord%20recognition%20network%20with%20Convolutional%20layers.ipynb)
  3. [Example 3: Setup `config.json`](https://github.com/keunwoochoi/dl4mir/blob/master/Example_3.py)
  4. [Example 4: download and preprocess](https://github.com/keunwoochoi/dl4mir/blob/master/Example_4.py)
  5. Real examples with real datasets!
    * [Example 5-1: Time-varying classification example using Jamendo dataset](https://github.com/keunwoochoi/dl4mir/blob/master/Example_5-1.py)
    * [Example 5-2: Time-invariant classification example using FMA dataset](https://github.com/keunwoochoi/dl4mir/blob/master/Example_5-2.py)

## Prerequisites
   ```
   $ pip install -r requirements.txt
   $ git clone https://github.com/keunwoochoi/kapre.git
   $ cd kapre
   $ python setup.py install
   ```
   to install
  * Librosa, Keras, Numpy, Matplotlib, Future
  * kapre

## Some links
  * [DL_MIR_TUTORIAL](https://github.com/tuwien-musicir/DL_MIR_Tutorial): Another DL for MIR tutorial by Thomas Lidy 
  * [DL in music informatics](http://steinhardt.nyu.edu/marl/research/deep_learning_in_music_informatics): ISMIR 2014 tutorial
  * [cs213n](http://cs231n.stanford.edu): Perhaps the most popular convnet tutorial. From Stanford university.
  * [librosa tutorial](https://librosa.github.io/librosa/tutorial.html): If you're interested in learning a bit more of MIR and its implementations.
  * [MIR datasets](https://www.audiocontentanalysis.org/data-sets/): An awesome list of MIR datasets
  * [/r/musicir](https://www.reddit.com/r/musicir/): A subreddit for MIR

## Cite?
```
@article{choi2017tutorial,
  title={A Tutorial on Deep Learning for Music Information Retrieval},
  author={Choi, Keunwoo and Fazekas, Gy{\"o}rgy and Cho, Kyunghyun and Sandler, Mark},
  journal={arXiv:1709.04396},
  year={2017}
}
```
Or visit [the paper page on Google scholar](https://scholar.google.co.kr/citations?view_op=view_citation&hl=en&user=ZrqdSu4AAAAJ&sortby=pubdate&citation_for_view=ZrqdSu4AAAAJ:W5xh706n7nkC) for potential updates.
