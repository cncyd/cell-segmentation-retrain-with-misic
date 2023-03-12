# cell-segmentation-retrain-with-misic
<p class="has-line-data" data-line-start="2" data-line-end="3">This is a workflow to retrain MISIC(Panigrahi S, Murat D, Le Gall A, et al. MISIC, a general deep learning-based method for the high-throughput cell segmentation of complex bacterial communities[J]. Elife, 2021, 10: e65151.) with your dataset.</p>

<p class="has-line-data" data-line-start="4" data-line-end="5"><a href="https://github.com/cncyd/cell-segmentation-retrain-with-misic/blob/main/misic_pro.yaml">misic_pro.yaml</a> gives the environment needed for the training.</p>

<p class="has-line-data" data-line-start="6" data-line-end="7"><a href="https://github.com/cncyd/cell-segmentation-retrain-with-misic/blob/main/my_class.py">my_class.py</a> defines a data generator that helps fetch a pair of target-label as a batch, as an u-net model recommends batch size to be 1 and somehow MISIC always reports errors when I tried to use batch size greater than 1.</p>

<p class="has-line-data" data-line-start="8" data-line-end="9"><a href="https://github.com/cncyd/cell-segmentation-retrain-with-misic/blob/main/trywithblank.py">trywithblank.py</a>  retrains MISIC with a weighted loss and Adam optimizer(ratio=0.001) and initializes weight and bias before training. This is a relatively more efficient method for my dataset. The hyperparameter used for normalization in weighted loss is determined by counting the proportion of 1s in the sparse matrix as ground truth. Without initialization, training quickly falls into bad local optima. Since I am a rookie in machine learning, this training method may be further optimized. You can customize it base on the characteristics of your dataset.</p>

<p class="has-line-data" data-line-start="10" data-line-end="11">Using the recognizable cell fluorescence images to make ground truth and then training to recognize its brightfield images is the innovation of this workflow. <a href="https://github.com/cncyd/cell-segmentation-retrain-with-misic/blob/main/resetvalue.py">resetvalue.py</a> makes labels using fluorescence images, and <a href="https://github.com/cncyd/cell-segmentation-retrain-with-misic/blob/main/makedata.py">makedata.py</a> 
makes targets with some preprocessing using brightfield images.</p>
