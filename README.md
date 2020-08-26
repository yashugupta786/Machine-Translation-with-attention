<html>
  <head>
    <title>Visualizing A Neural Machine Translation  Model with Attention Mechanism</title>

Author:Yashu GUPTA.

Note: The animations below are videos. Touch or hover on them (if you’re using a mouse) to get play controls so you can pause if needed.

Sequence-to-sequence models are deep learning models that have achieved a lot of success in tasks like machine translation, text summarization, and image captioning. Google Translate started using such a model in production in late 2016. These models are explained in the two pioneering papers (Sutskever et al., 2014, Cho et al., 2014).

I found, however, that understanding the model well enough to implement it requires unraveling a series of concepts that build on top of each other. I thought that a bunch of these ideas would be more accessible if expressed visually. That’s what I aim to do in this post. You’ll need some previous understanding of deep learning to get through this post. I hope it can be a useful companion to reading the papers mentioned above (and the attention papers linked later in the post).

A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images…etc) and outputs another sequence of items. A trained model would work like this:

  
  Your browser does not support the video tag.


" />
    <meta property="og:description" content="Translations: Chinese (Simplified), Korean

Watch: MIT’s Deep Learning State of the Art lecture referencing this post



Note: The animations below are videos. Touch or hover on them (if you’re using a mouse) to get play controls so you can pause if needed.

Sequence-to-sequence models are deep learning models that have achieved a lot of success in tasks like machine translation, text summarization, and image captioning. Google Translate started using such a model in production in late 2016. These models are explained in the two pioneering papers (Sutskever et al., 2014, Cho et al., 2014).

I found, however, that understanding the model well enough to implement it requires unraveling a series of concepts that build on top of each other. I thought that a bunch of these ideas would be more accessible if expressed visually. That’s what I aim to do in this post. You’ll need some previous understanding of deep learning to get through this post. I hope it can be a useful companion to reading the papers mentioned above (and the attention papers linked later in the post).

A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images…etc) and outputs another sequence of items. A trained model would work like this:

  
  Your browser does not support the video tag.


" />
    
 

    
    <meta property="og:title" content="Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)" />
    <meta property="twitter:title" content="Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)" />
    

    <!--[if lt IE 9]>
      <script src="http://html5shiv.googlecode.com/svn/trunk/html5.js"></script>
    <![endif]-->

    <script src="/js/jquery-3.1.1.slim.min.js"></script>
    <script type="text/javascript" src="/js/d3.min.js"></script>
    <script type="text/javascript" src="/js/d3-selection-multi.v0.4.min.js"></script>
    <script type="text/javascript" src="/js/d3-jetpack.js"></script>

    <link rel="stylesheet" href="/css/bootstrap.min.css" />
    <link rel="stylesheet" href="/css/bootstrap-theme.min.css" />
    <script src="/js/bootstrap.min.js"> </script>

    <link rel="stylesheet" type="text/css" href="/bower_components/jquery.gifplayer/dist/gifplayer.css" />
    <script type="text/javascript" src="/bower_components/jquery.gifplayer/dist/jquery.gifplayer.js"></script>

    <!--
    <script data-main="scripts/main" src="scripts/require.js"></script>
    -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.css" integrity="sha384-wE+lCONuEo/QSfLb4AfrSk7HjWJtc4Xc1OiB2/aDBzHzjnlBP4SX7vjErTcwlA8C" crossorigin="anonymous" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.6.0/katex.min.js" integrity="sha384-tdtuPw3yx/rnUGmnLNWXtfjb9fpmwexsd+lr6HUYnUY4B7JhB5Ty7a1mYd+kto/s" crossorigin="anonymous"></script>

    <link rel="stylesheet" type="text/css" href="/style.css" />
   

    <meta name="viewport" content="width=device-width" />
    <!-- Created with Jekyll Now - http://github.com/barryclark/jekyll-now -->

    <!-- Piwik -->
    <!-- Piwik
    <script type="text/javascript">
        var _paq = _paq || [];
        _paq.push(["setDomains", ["*.example.org"]]);
        _paq.push(['trackPageView']);
        _paq.push(['enableLinkTracking']);
        (function() {
            var u="https://a.jalammar.com/";
            _paq.push(['setTrackerUrl', u+'piwik.php']);
            _paq.push(['setSiteId', '1']);
            var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
            g.type='text/javascript'; g.async=true; g.defer=true; g.src=u+'piwik.js'; s.parentNode.insertBefore(g,s);
        })();
    </script>
    <noscript><p><img src="https://a.jalammar.com/piwik.php?idsite=1" style="border:0;" alt="" /></p></noscript>-->
    <!-- End Piwik Code -->

    <!-- End Piwik Code -->
  </head>

  <body>
    
    <div id="main" role="main" class="container">
      <article class="post">
  <h1>Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)</h1>

  <div class="entry prediction">
    <p><span class="discussion">Translations: <a href="https://blog.csdn.net/qq_41664845/article/details/84245520">Chinese (Simplified)</a>, <a href="https://nlpinkorean.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/">Korean</a></span>
<br />
<span class="discussion">Watch: MIT’s <a href="https://youtu.be/53YvP6gdD7U?t=335">Deep Learning State of the Art</a> lecture referencing this post</span></p>


<p><strong>Note:</strong> The animations below are videos. Touch or hover on them (if you’re using a mouse) to get play controls so you can pause if needed.</p>

<p>Sequence-to-sequence models are deep learning models that have achieved a lot of success in tasks like machine translation, text summarization, and image captioning. Google Translate started <a href="https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/">using</a> such a model in production in late 2016. These models are explained in the two pioneering papers (<a href="https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf">Sutskever et al., 2014</a>, <a href="http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf">Cho et al., 2014</a>).</p>

<p>I found, however, that understanding the model well enough to implement it requires unraveling a series of concepts that build on top of each other. I thought that a bunch of these ideas would be more accessible if expressed visually. That’s what I aim to do in this post. You’ll need some previous understanding of deep learning to get through this post. I hope it can be a useful companion to reading the papers mentioned above (and the attention papers linked later in the post).</p>

<p>A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images…etc) and outputs another sequence of items. A trained model would work like this:</p>
<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="seq2seq_1.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<!--more-->

<p><br /></p>

<p>In neural machine translation, a sequence is a series of words, processed one after another. The output is, likewise, a series of words:</p>

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="seq2seq_2.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<h2 id="looking-under-the-hood">Looking under the hood</h2>

<p>Under the hood, the model is composed of an <span class="encoder">encoder</span> and a <span class="decoder">decoder</span>.</p>

<p>The <span class="encoder">encoder</span> processes each item in the input sequence, it compiles the information it captures into a vector (called the <span class="context">context</span>). After processing the entire input sequence, the <span class="encoder">encoder</span> send the <span class="context">context</span>  over to the <span class="decoder">decoder</span>, which begins producing the output sequence item by item.</p>

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="seq2seq_3.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<p><br /></p>

<p>The same applies in the case of machine translation.</p>

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="seq2seq_4.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<p>The <span class="context">context</span>  is a vector (an array of numbers, basically) in the case of machine translation. The <span class="encoder">encoder</span> and <span class="decoder">decoder</span>  tend to both be recurrent neural networks (Be sure to check out Luis Serrano’s <a href="https://www.youtube.com/watch?v=UNmqTiOnRfg">A friendly introduction to Recurrent Neural Networks</a> for an intro to RNNs).</p>

<div class="img-div">
    <img src="context.png" />
    The <span class="context">context</span>  is a vector of floats. Later in this post we will visualize vectors in color by assigning brighter colors to the cells with higher values.
</div>

<p>You can set the size of the <span class="context">context</span>  vector when you set up your model. It is basically the number of hidden units in the <span class="encoder">encoder</span> RNN. These visualizations show a vector of size 4, but in real world applications the <span class="context">context</span> vector would be of a size like 256, 512, or 1024.</p>

<p><br /></p>

<p>By design, a RNN takes two inputs at each time step: an input (in the case of the encoder, one word from the input sentence), and a hidden state. The word, however, needs to be represented by a vector. To transform a word into a vector, we turn to the class of methods called “<a href="https://machinelearningmastery.com/what-are-word-embeddings/">word embedding</a>” algorithms. These turn words into vector spaces that capture a lot of the meaning/semantic information of the words (e.g. <a href="http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html">king - man + woman = queen</a>).</p>

<p><br /></p>

<div class="img-div">
    <img src="embedding.png" />
    We need to turn the input words into vectors before processing them. That transformation is done using a <a href="https://en.wikipedia.org/wiki/Word_embedding">word embedding</a> algorithm. We can use <a href="http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/">pre-trained embeddings</a> or train our own embedding on our dataset. Embedding vectors of size 200 or 300 are typical, we're showing a vector of size four for simplicity.
</div>

<p>Now that we’ve introduced our main vectors/tensors, let’s recap the mechanics of an RNN and establish a visual language to describe these models:</p>

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="RNN_1.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<p><br /></p>

<p>The next RNN step takes the second input vector and hidden state #1 to create the output of that time step. Later in the post, we’ll use an animation like this to describe the vectors inside a neural machine translation model.</p>

<p><br /></p>

<p>In the following visualization, each pulse for the <span class="encoder">encoder</span> or <span class="decoder">decoder</span>  is that RNN processing its inputs and generating an output for that time step. Since the <span class="encoder">encoder</span> and <span class="decoder">decoder</span>  are both RNNs, each time step one of the RNNs does some processing, it updates its <span class="context">hidden state</span>  based on its inputs and previous inputs it has seen.</p>

<p>Let’s look at the <span class="context">hidden states</span>  for the <span class="encoder">encoder</span>. Notice how the last <span class="context">hidden state</span>  is actually the <span class="context">context</span>  we pass along to the <span class="decoder">decoder</span>.</p>

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="seq2seq_5.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<p><br /></p>

<p>The <span class="decoder">decoder</span>  also maintains a <span class="decoder">hidden states</span>  that it passes from one time step to the next. We just didn’t visualize it in this graphic because we’re concerned with the major parts of the model for now.</p>

<p>Let’s now look at another way to visualize a sequence-to-sequence model. This animation will make it easier to understand the static graphics that describe these models. This is called an “unrolled” view where instead of showing the one <span class="decoder">decoder</span>, we show a copy of it for each time step. This way we can look at the inputs and outputs of each time step.</p>

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="seq2seq_6.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<p><br /></p>

<h1 id="lets-pay-attention-now">Let’s Pay Attention Now</h1>
<p>The <span class="context">context</span>  vector turned out to be a bottleneck for these types of models. It made it challenging for the models to deal with long sentences. A solution was proposed in <a href="https://arxiv.org/abs/1409.0473">Bahdanau et al., 2014</a> and <a href="https://arxiv.org/abs/1508.04025">Luong et al., 2015</a>. These papers introduced and refined a technique called “Attention”, which highly improved the quality of machine translation systems. Attention allows the model to focus on the relevant parts of the input sequence as needed.</p>

<p><img src="attention.png" /></p>

<div class="img-div">
    At time step 7, the attention mechanism enables the <span class="decoder">decoder</span>  to focus on the word "étudiant" ("student" in french) before it generates the English translation. This ability to amplify the signal from the relevant part of the input sequence makes attention models produce better results than models without attention.
</div>

<p><br /></p>

<p>Let’s continue looking at attention models at this high level of abstraction. An attention model differs from a classic sequence-to-sequence model in two main ways:</p>

<p>First, the <span class="encoder">encoder</span> passes a lot more data to the <span class="decoder">decoder</span>. Instead of passing the last hidden state of the encoding stage, the <span class="encoder">encoder</span> passes <em>all</em> the <span class="context">hidden states</span>  to the <span class="decoder">decoder</span>:</p>

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="seq2seq_7.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<p><br /></p>

<p>Second, an attention <span class="decoder">decoder</span>  does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the <span class="decoder">decoder</span>  does the following:</p>

<ol>
  <li>Look at the set of encoder <span class="context">hidden states</span>  it received – each <span class="context">encoder hidden states</span>  is most associated with a certain word in the input sentence</li>
  <li>Give each <span class="context">hidden states</span>  a score (let’s ignore how the scoring is done for now)</li>
  <li>Multiply each <span class="context">hidden states</span>  by its softmaxed score, thus amplifying <span class="context">hidden states</span>  with high scores, and drowning out <span class="context">hidden states</span>  with low scores</li>
</ol>

<video width="100%" height="auto" loop="" autoplay="" controls="">
   <source src="attention_process.mp4" type="video/mp4" />
   Your browser does not support the video tag.
</video>

<p><br />
<br /></p>

<p>This scoring exercise is done at each time step on the <span class="decoder">decoder</span> side.</p>

<p>Let us now bring the whole thing together in the following visualization and look at how the attention process works:</p>

<ol>
  <li>The attention decoder RNN takes in the embedding of the <span class="embedding">&lt;END&gt;</span> token, and an <span class="decoder">initial decoder hidden state</span>.</li>
  <li>The RNN processes its inputs, producing an output and a <span class="decoder">new hidden state</span> vector (<span class="decoder">h</span><span class="step_no">4</span>). The output is discarded.</li>
  <li>Attention Step: We use the <span class="context">encoder hidden states</span> and the <span class="decoder">h</span><span class="step_no">4</span> vector to calculate a context vector (<span class="step_no">C</span><span class="decoder">4</span>) for this time step.</li>
  <li>We concatenate <span class="decoder">h</span><span class="step_no">4</span> and <span class="step_no">C</span><span class="decoder">4</span> into one vector.</li>
  <li>We pass this vector through a <span class="ffnn">feedforward neural network</span> (one trained jointly with the model).</li>
  <li>The <span class="logits_output">output</span> of the feedforward neural networks indicates the output word of this time step.</li>
  <li>Repeat for the next time steps</li>
</ol>

<video width="100%" height="auto" loop="" autoplay="" controls="">
   <source src="attention_tensor_dance.mp4" type="video/mp4" />
   Your browser does not support the video tag.
</video>

<p><br />
<br /></p>

<p>This is another way to look at which part of the input sentence we’re paying attention to at each decoding step:</p>

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="seq2seq_9.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<p>Note that the model isn’t just mindless aligning the first word at the output with the first word from the input. It actually learned from the training phase how to align words in that language pair (French and English in our example). An example for how precise this mechanism can be comes from the attention papers listed above:</p>

<div class="img-div">
<img src="attention_sentence.png" />
    You can see how the model paid attention correctly when outputing "European Economic Area". In French, the order of these words is reversed ("européenne économique zone") as compared to English. Every other word in the sentence is in similar order.
</div>

<p><br /></p>

<p>If you feel you’re ready to learn the implementation, be sure to check TensorFlow’s <a href="https://github.com/tensorflow/nmt">Neural Machine Translation (seq2seq) Tutorial</a>.</p>
<h1>Github Link <a href="https://github.com/yashugupta786">Yashu Github link </a></h1>
<p>Source post all credits goes to Jalammar <a href="https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention">NMT Jalammar post</a></p>
<hr />

<p><br /></p>




    <div class="wrapper-footer">
      <div class="container">
        <footer class="footer">
          
       </footer>
      </div>
    </div>

    

      <footer class="site-footer">
        
          <span class="site-footer-owner"><a href="https://github.com/yashugupta786/yashu.github.io">yashu.github.io</a> is maintained by <a href="https://github.com/yashugupta786">yashugupta786</a>.</span>
      </footer>
